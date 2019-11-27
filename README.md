# Android Face Recognize with JavaCV 

Android app using JavaCV to face recognize.<br><br>
Project based on example <b><a href="https://github.com/bytedeco">Bytedeco's</a></b> project: <br>
```https://github.com/bytedeco/sample-projects/tree/master/javacv-android-recognize/app/src/main/java/org/bytedeco/javacv/android/recognize```

Used recognize algorithms:
--------------------------
 * EigenFaces
 * FisherFaces
 * LBPH

App allows to:
--------------
 * Manage users
 * Taking photos
 * Face detection (from camera and on photos)
 * Training algorithms
 * Recognizing from camera
 * Recognizing from photos
 * Visualisation of Eigen faces in all steps of training

Gradle (inside the `build.gradle` file):
-----------------------------------------
```implementation group: 'org.bytedeco', name: 'javacv', version: '1.4.3'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.3-1.4.3', classifier: 'android-arm'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'opencv', version: '3.4.3-1.4.3', classifier: 'android-x86'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'ffmpeg', version: '4.0.2-1.4.3', classifier: 'android-arm'
    implementation group: 'org.bytedeco.javacpp-presets', name: 'ffmpeg', version: '4.0.2-1.4.3', classifier: 'android-x86'```