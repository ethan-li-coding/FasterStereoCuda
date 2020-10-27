#ifndef SGM_STEREO_TYPES_H_
#define SGM_STEREO_TYPES_H_

#include <vector>
using std::vector;

#ifndef SGM_SAFE_DELETE
#define SGM_SAFE_DELETE
#define SafeFree(ptr) if(ptr) {_mm_free(ptr); ptr = nullptr;}
#define SafeDeleteArray(ptr) if(ptr) {delete[] ptr; ptr = nullptr;}
#define SafeDelete(ptr) if(ptr) {delete ptr; ptr = nullptr;}
#endif

#endif //SGM_STEREO_TYPES_H_