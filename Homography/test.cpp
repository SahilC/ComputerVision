Mat my_segment(Mat _inImage, Rect assumption, Rect face){ 
// human segmentation opencv

        // _inImage - input image
        // assumption - human rectangle on _inImage
        // face - face rectangle on _inImage
        // iterations  - is being set externally


        /* 
        GrabCut segmentation        
        */


        Mat bgModel,fgModel; // the models (internally used)
        Mat result; // segmentation result


        //*********************** step1: GrabCut human figure segmentation
        grabCut(_inImage,    // input image
            result,   // segmentation result
            assumption,// rectangle containing foreground
            bgModel,fgModel, // models
            iterations,        // number of iterations
            cv::GC_INIT_WITH_RECT); // use rectangle

        // Get the pixels marked as likely foreground
        cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);

        // upsample the resulting mask       

        cv::Mat separated(assumption.size(),CV_8UC3,cv::Scalar(255,255,255));
        _inImage(assumption).copyTo(separated,result(assumption));  
         // (bg pixels not copied)

        //return(separated); // return the innerings of assumption rectangle


        //*********************** step2: 
        //cutting the skin with the mask based on the face rectangle
        Rect adjusted_face = face;
        adjusted_face.x = face.x - assumption.x;
        adjusted_face.y = face.y - assumption.y;

        //rectangle(separated,  
        //     adjusted_face.tl(), 
        //     adjusted_face.br(), 
        //     cv::Scalar(166,94,91), 2);  

        //creating mask
        Mat mymask(separated.size(),CV_8UC1);  

        // setting face area as sure background
        mymask.setTo(Scalar::all(GC_PR_FGD));
        mymask(adjusted_face).setTo(Scalar::all(GC_BGD));

        // performing grabcut
        grabCut(separated,
           mymask,
           cv::Rect(0,0,assumption.width,assumption.height),
           bgModel,fgModel,
           1,
           cv::GC_INIT_WITH_MASK);

        // Just repeating everything from before

        // Get the pixels marked as likely foreground
        cv::compare(mymask,cv::GC_PR_FGD,mymask,cv::CMP_EQ);

        //here was the error
        //separated.copyTo(separated,mymask);  // bg pixels not copied

        cv::Mat res(separated.size(),CV_8UC3,cv::Scalar(255,255,255));
        separated.copyTo(res,mymask);  // bg pixels not copied

        //*************************//   

        //return(separated); // return the innerings of assumption rectangle
        return(res);            

} 