#include "sparse_sampler.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <string>
//#include "lda_option.hpp"
#include <iostream>

namespace LDA{

	SparseSampler::SparseSampler(int32_t _V, const std::vector<int>& _topic_summary_table, const std::vector<std::unordered_map<int32_t, int32_t>>& _word_topic_table){
		//std::random_device rd;
		//rng_engine.reset(new std::mt19937(rd()));

		// Obtain Topic Model Parameter
		//LDAOption ldaOption = LDAOption::getInstance();
		K = FLAGS_num_topics; // number of topics
		V = _V;	// number of vocabs		
		beta = FLAGS_beta;
		beta_sum = V * beta;
		alpha = FLAGS_alpha;

		topic_summary_table = _topic_summary_table;
		word_topic_table = _word_topic_table;
 
		q_coeff.resize(K);
		//nonzero_q_terms.resize(K);
		//nonzero_q_terms_topic.resize(K);
		doc_topic_vector.resize(K);
  		nonzero_doc_topic_idx = new int32_t[K];
		//nonzero_doc_topic_idx = new int32_t[K];
		//std::cout << "TEST2: " << K << std::endl;
	}

	SparseSampler::~SparseSampler(){
		//delete[] nonzero_doc_topic_idx;
	}

	void SparseSampler::sampleDoc(LDADoc& doc){
		//	before sampling every document
		//	we shall compute the doc_topic_vector for this doc
		computeDocTopic(doc);
		initializePrecomputableVariable();

		//	iterate over current document
		double alpha_beta = alpha * beta;
		//if (!doc)
		//	std::cout << "doc is NULL" <<std::endl;
		int32_t docSize = doc.getNumTokens();
		//int32_t docSize =10;
		for (int32_t d = 0; d < docSize; d++)
		{
			//	obtain current token's index and topic
			int32_t wordIdx = doc.getToken(d);
			int32_t old_topic = doc.getTokenTopic(d);
			//	Step 1:
			//	discard this "old_topic" in the sampling process
			//	Similar to initialization process
			double denominator = topic_summary_table[old_topic] + beta_sum;
			q_coeff[old_topic] = ( alpha + doc_topic_vector[old_topic] - 1 ) / ( denominator - 1 );
			s_sum -= alpha_beta / denominator;
			s_sum += alpha_beta / ( denominator - 1 );
			r_sum -= doc_topic_vector[old_topic] * beta / denominator;
			r_sum += ( doc_topic_vector[old_topic] - 1 ) * beta /  ( denominator - 1 );

			doc_topic_vector[old_topic]--;

			// if the topic count in this document goes to 0, update non-zero document topic 
		    if (doc_topic_vector[old_topic] == 0) {
		      int32_t* zero_idx =
		        std::lower_bound(nonzero_doc_topic_idx,
		            nonzero_doc_topic_idx + num_nonzero_doc_topic_idx, old_topic);
		      memmove(zero_idx, zero_idx + 1,
		          (nonzero_doc_topic_idx + num_nonzero_doc_topic_idx - zero_idx - 1)
		          * sizeof(int32_t));
		      --num_nonzero_doc_topic_idx;
		    }

			topic_summary_table[old_topic]--;

			//	Step 2:
			//	Compute q bucket mass

			//	obtain the topic count of this word from word topic table
			std::unordered_map<int32_t, int32_t> topic_count_row = word_topic_table[wordIdx];

			q_sum = 0;
  			nonzero_q_terms_topic.clear();

			for (std::unordered_map<int32_t, int32_t>::iterator it = topic_count_row.begin(); 
				it != topic_count_row.end(); it++)
			{
				int32_t current_topic = it->first;
				int32_t current_word_count = it->second;
				double temp_q_term = 0;
				if (current_topic == old_topic)
					temp_q_term = q_coeff[current_topic] * ( current_word_count - 1 );
				else
					temp_q_term = q_coeff[current_topic] * current_word_count;
				q_sum += temp_q_term;
				QTermTopic qTermTopic;
				qTermTopic.nonzero_q_term = temp_q_term;
				qTermTopic.topic = current_topic;
				nonzero_q_terms_topic.push_back(qTermTopic);
			}
			


			//	Step 3:
			//	Start Sampling
			int32_t new_topic = sampleTopic();
			//	Step 4:
			//	Update statistics for the new topic assignment
			//	This step is simliar to Step 1 except that the new topic is added to the statistics
			denominator = topic_summary_table[new_topic] + beta_sum;
			q_coeff[new_topic] = ( alpha + doc_topic_vector[new_topic] + 1 ) / ( denominator + 1 );
			s_sum -= alpha_beta / denominator;
			s_sum += alpha_beta / ( denominator + 1 );
			r_sum -= doc_topic_vector[new_topic] * beta / denominator;
			r_sum += ( doc_topic_vector[new_topic] + 1 ) * beta /  ( denominator + 1 );

			doc_topic_vector[new_topic]++;

			// if the topic count in this document goes to 1 after sampling, update non-zero document topic list
			if (doc_topic_vector[new_topic] == 1) {
		      int32_t* insert_idx =
		        std::lower_bound(nonzero_doc_topic_idx,
		            nonzero_doc_topic_idx + num_nonzero_doc_topic_idx, new_topic);
		      memmove(insert_idx + 1, insert_idx, (nonzero_doc_topic_idx +
		            num_nonzero_doc_topic_idx - insert_idx) * sizeof(int32_t));
		      *insert_idx = new_topic;
		      ++num_nonzero_doc_topic_idx;
		    }

			//	Step 5:
			//	Finally, update the topic of this token in this document, summary table and word topic table
			
			//doc.setTokenTopic(wordIdx, new_topic);
			doc.setTokenTopic(d, new_topic);
			topic_summary_table[new_topic]++;

			word_topic_table[wordIdx][new_topic] += 1;
			word_topic_table[wordIdx][old_topic] -= 1;

			//std::cout << "word index: [" << d << "] old_topic: [" << old_topic << "] new_topic: [" << new_topic << "]" << std::endl;
		
		}
	}

	int32_t SparseSampler::sampleTopic(){
		double total_mass = s_sum + r_sum + q_sum;
		std::random_device rd;
		std::mt19937 rng_engine(rd());
		std::uniform_real_distribution<> dis(0, total_mass);
		
		double random_outcome = dis(rng_engine);

		if ( random_outcome < s_sum ) // fall into "smoothing only" bucket
		{
			double local_s_sum = 0;
			for ( int32_t k = 0; k < K; k++)
			{
				double current_s_term = ( alpha * beta ) / ( topic_summary_table[k] + beta_sum );
				local_s_sum += current_s_term;
				if (random_outcome >  local_s_sum)
					return k;
			}
			return K - 1; 	//	handle case random_outcome == s_sum
		}
		else 
		{
			random_outcome = random_outcome - s_sum;
			if ( random_outcome < r_sum ) // fall into "doc topic" bucket
			{
				double local_r_sum = 0;
				int32_t current_topic = 0;
				for (int i = 0; i < num_nonzero_doc_topic_idx; ++i) {
			        int32_t current_topic = nonzero_doc_topic_idx[i];
			        local_r_sum += doc_topic_vector[current_topic] * beta / ( topic_summary_table[current_topic] + beta_sum);
			        if ( random_outcome > local_r_sum )
						return current_topic;
			    }
			    return current_topic;
			    /*
				// iterate over nonzero_doc_topic_idx list
				// since number of non zero document topics <<<<<< number of total topics
				double local_r_sum = 0;
				int32_t current_topic = 0;
				for (std::list<int32_t>::iterator it = nonzero_doc_topic_idx.begin(); it != nonzero_doc_topic_idx.end(); it++)
				{
					current_topic = (*it);
					local_r_sum += doc_topic_vector[current_topic] * beta / ( topic_summary_table[current_topic] + beta_sum);
					if ( random_outcome > local_r_sum )
						return current_topic;
				}
				return current_topic;	// similarly, handle case random_outcome == r_sum
				*/
			}
			else	// fall into "word topic" bucket
			{
				random_outcome = random_outcome - r_sum;
				double local_q_sum = 0;
				int32_t current_topic = 0;
				for ( std::list<QTermTopic>::iterator it = nonzero_q_terms_topic.begin(); it != nonzero_q_terms_topic.end(); it++)
				{
					double current_nonzero_q_term = it->nonzero_q_term;
					current_topic = it->topic;
					local_q_sum += current_nonzero_q_term;
					if ( random_outcome > local_q_sum )
						return current_topic;
				}
				return current_topic;	// similarly, handle case random_outcome == q_sum
			}
			
		}
	}

	void SparseSampler::computeDocTopic(LDADoc doc){
		std::fill(doc_topic_vector.begin(), doc_topic_vector.end(), 0);
		int32_t docSize = doc.getNumTokens();
		for (int32_t i = 0; i < docSize; i++){
			int32_t tokenTopic = doc.getTokenTopic(i);
			doc_topic_vector[tokenTopic]++;
		}
	}

	//	By ths function, we can obtain
	//	1.	"smoothing only" bucket
	//	2.	"document topic" bucket
	//	3.	q coeff of "topic word" bucket
	void SparseSampler::initializePrecomputableVariable(){
		num_nonzero_doc_topic_idx = 0;
		s_sum = 0;
		r_sum = 0;
		double alpha_beta = alpha * beta;

		for (int32_t k = 0; k < K; k++)
		{
			double denominator = topic_summary_table[k] + beta_sum;
			q_coeff[k] = ( alpha + doc_topic_vector[k] ) / denominator;
			s_sum += alpha_beta /  denominator;
			//r_sum += doc_topic_vector[k] * beta / denominator;
			if (doc_topic_vector[k] != 0)
			{
				r_sum += doc_topic_vector[k] * beta / denominator;
				nonzero_doc_topic_idx[num_nonzero_doc_topic_idx++] = k;			
			}
		}
	}

}
