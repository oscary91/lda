#include "gflags/gflags.h"
#include "glog/logging.h"
#include "test1.hpp"
namespace flexps{
	
	Test123::Test123(){
		LOG(INFO) << "Hello Test!";
	}	
/*	
	class Test{
	public:
		Test(){
		LOG(INFO) << "Hello Test!";
		}	
	};
*/
}
