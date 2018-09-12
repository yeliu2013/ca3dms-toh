#include "cv.h"
#include "highgui.h"
#include <vector>
#include <fstream>
#include "Tracker.h"
#include "time.h"
#include <opencv2/objdetect/objdetect.hpp> 
using namespace cv;

int main()
{
	Tracker t;
	t.track_all();
}