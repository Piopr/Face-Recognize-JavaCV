# Android Face Recognize with JavaCV 

Android app using JavaCV to face recognize.<br><br>
Project based on example <b><a href="https://github.com/bytedeco">Bytedeco's</a></b> project: <br>
```https://github.com/bytedeco/sample-projects/tree/master/javacv-android-recognize/app/src/main/java/org/bytedeco/javacv/android/recognize```

Used recognize algorithms:
--------------------------
 * EigenFaces
 * FisherFaces
 * LBPH
 
 Screenshots:
-----------------------------------------

Main menu screenshot and details
    
![mainmenu](https://user-images.githubusercontent.com/32546106/73701894-4cebc480-46eb-11ea-83e4-cbde73d48d84.png)

<details>
   <summary>Details</summary>
 <b>Add</b> - adding new user (string name in text field needed.</br>
    <b>Delete</b> - deletes the selected user.  </br>
    <b>Recognize</b> -  switch to recognize activity.
 
 </details>


Recognize activity and details:
Before training: 
    
![beforetraining](https://user-images.githubusercontent.com/32546106/73701898-4e1cf180-46eb-11ea-84fb-d8fd780123b9.png)
    
After training:

![aftertraining](https://user-images.githubusercontent.com/32546106/73701900-4f4e1e80-46eb-11ea-9513-d8cbe0dd09df.png)

<details>
 <summary>Details</summary>
 <b>How to take a correct photo</b></br>
        Be sure that your face is in the square (correctly detected).</br>
        <b>Important things:</b> </br>
        <b>light - </b> make sure your face is evenly lit. Avoid situations where the face is difficult to see or only part of your face is lit.</br>
        <b>background - </b> your background should be uniform.</br>
        <b>face position - </b> the face should be in a vertical position (eyes should be in one horizontal line).</br>
        <b>number of photos - </b> for good results number of photos should be higher than 7.</br>
        <b> Not following the rules might affect bad results!</b></br></br>
        <b>Interface</b></br>
       <b>Take photo</b> - Take photo ready to train.</br>
        <b>Train</b> - Algorithms training. (with your own dataset first click <b>rename</b> and then <b>det. photo</b></br>
        <b>Reset</b> - reset of trained algorithms.</br>
        <b>Rename</b> - changing a photo names to pattern person.id.photo_number.jpg </br>
        <b>Det. Photo</b> - detect faces on dataset of current user and resize to correct resolution. </br>
        <b>Recog. photo</b> - recognize user from photo of current user in folder <i>default</i>. </br>
        <b>Mean</b> - performs visualization of selected face recognition algorithms available in <b> user/visualizations </b>. </br>
        <b>Test</b> - performs a recognition test on all photo sets. Results in .csv files in the main folder.
 
 </details>


App allows to:
--------------
 * Manage users
 * Taking photos
 * Face detection (from camera and on photos)
 * Training algorithms
 * Recognizing from camera
 * Recognizing from photos
 * Visualisation of Eigen faces in all steps of training
 * Time comparing of all algorithms

Gradle (inside the `build.gradle` file):
-----------------------------------------
```implementation group: 'org.bytedeco', name: 'javacv', version: '1.4.3'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.3-1.4.3', classifier: 'android-arm'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.3-1.4.3', classifier: 'android-x86'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'ffmpeg', version: '4.0.2-1.4.3', classifier: 'android-arm'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'ffmpeg', version: '4.0.2-1.4.3', classifier: 'android-x86'
```

Requirements:
-----------------------------------------
 * min. Android 4.1 or upper
 * Camera 
 

    



