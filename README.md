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
 <details>
 <summary>Main menu screenshot and details</summary>
    
![mainmenu](https://user-images.githubusercontent.com/32546106/73701894-4cebc480-46eb-11ea-83e4-cbde73d48d84.png)

</details>

<details>
 <summary>Recognize activity and details:</summary>
Before training: 
    
![beforetraining](https://user-images.githubusercontent.com/32546106/73701898-4e1cf180-46eb-11ea-84fb-d8fd780123b9.png)
    
After training:

![aftertraining](https://user-images.githubusercontent.com/32546106/73701900-4f4e1e80-46eb-11ea-9513-d8cbe0dd09df.png)

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
 

    



