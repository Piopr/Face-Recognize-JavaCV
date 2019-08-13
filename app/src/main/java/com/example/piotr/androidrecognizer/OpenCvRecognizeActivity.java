package com.example.piotr.androidrecognizer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.View;
import android.widget.Toast;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;
import java.io.File;

import static com.example.piotr.androidrecognizer.TrainHelper.TRAIN_FOLDER;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.cvRound;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static com.example.piotr.androidrecognizer.TrainHelper.ACCEPT_LEVEL;

/**
 * Created by djalmaafilho.
 */
public class OpenCvRecognizeActivity extends Activity implements CvCameraPreview.CvCameraViewListener {
    public static final String TAG = "OpenCvRecognizeActivity";
    /**
     * zmienna CascadeClassifiera, czyli wzorca do detekcji twarzy
     */
    private CascadeClassifier faceDetector;
    /**
     * Komunikat pojawiający się podczas wykrycia twarzy
     */
    private String[] nomes = {"", "Y Know You"};
    /**
     * minimalny rozmiar twarzy na activity kamery. Minimalna wartość, to 1/3 szerokości activity.
     */
    private int absoluteFaceSize = 0;
    /**
     * obiekt do sterowania kamerą
     */
    private CvCameraPreview cameraView;
    /**
     * gdy zdjęcie jest wykonywane - true
     * gdy nie jest lub zakońcono zmieniana na - false
     */
    boolean takePhoto;

    /**
     * obiekt do rozpoznawania twarzy
     */
    opencv_face.FaceRecognizer faceRecognizer = opencv_face.EigenFaceRecognizer.create();

    /**
     * jesli już nauczono twarzy (istnieje plik .yml) zmienia się na true i można zacząć rozpoznawanie twarzy
     */
    boolean trained;

    /**
     *
     * metoda sprawdzania uprawnien. Zwraca true gdy wszystkie wymagane uprawnienia są nadane
     */
    private boolean hasPermissions(Context context, String... permissions) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }
//
    /*
    Przy tworzeniu activity
     */
    @SuppressLint("StaticFieldLeak")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencv);
        /*
        Sprawdzanie uprawnien czytania i pisania
         */
        if (Build.VERSION.SDK_INT >= 23) {
            String[] PERMISSIONS = {android.Manifest.permission.READ_EXTERNAL_STORAGE,android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
            if (!hasPermissions(this, PERMISSIONS)) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, 1 );
            }
        }
        /*
        wczytanie z activity zmiennej obsługującej podgląd kamery (wykrywanie twarzy itd.)
         */
        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);


        /*
        wykonywanie operacji w tle
         */
        new AsyncTask<Void,Void,Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    /***
                     * wczytanie wzorca do detekcji twarzy
                     */
                    faceDetector = TrainHelper.loadClassifierCascade(OpenCvRecognizeActivity.this, R.raw.frontalface);
                    /***
                     * Sprawdzenie, czy algorytm jest już nauczony zestawem zdjęć
                     * Zdjęcia znajdują się w lokalizacji *\TRAIN_FOLDER
                     * TODO: zmiana na / *TRAIN_FOLDER/user1, user2 itd.
                     *
                     * na razie tylko do obsłubi EIGEN_FACES
                     */
                    if(TrainHelper.isTrained(getBaseContext())) {
                        //File folder = new File(getFilesDir(), TrainHelper.TRAIN_FOLDER);
                        File folder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
                        File f = new File(folder, TrainHelper.EIGEN_FACES_CLASSIFIER);
//                        faceRecognizer.load(f.getAbsolutePath());
                        faceRecognizer.read(f.getAbsolutePath());
                        trained = true;
                    }
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            /***
             *
             * Po zakończeniu sprawdzania nauczenia algorytmu
             * Obsługa przycisków
             *
             */
            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                findViewById(R.id.btPhoto).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        takePhoto = true;
                    }
                });
                findViewById(R.id.btTrain).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        train();
                    }
                });
                /***
                 * Po zakończeniu powrót do poprzedniego acitvity
                 */
                findViewById(R.id.btReset).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            TrainHelper.reset(getBaseContext());
                            Toast.makeText(getBaseContext(), "Reseted with sucess.", Toast.LENGTH_SHORT).show();
                            finish();
                        }catch (Exception e) {
                            Log.d(TAG, e.getLocalizedMessage(), e);
                        }
                    }
                });
            }
        }.execute();
    }

    /***
     * Metoda do treningu algorytmu
     * 1. Sprawdzenie ilości wykonanych zdjęć (alerty gdy za mało lub gdy już wytrenowano algorytm)
     * 2. Jeśli oba warunki niespełnione start uczenia algorytmu.
     * 3. Komunikat o powodzeniu nauczenia
     */
    @SuppressLint("StaticFieldLeak")
    void train() {
        int remainigPhotos = TrainHelper.PHOTOS_TRAIN_QTY - TrainHelper.qtdPhotos(getBaseContext());
        if(remainigPhotos > 0) {
            Toast.makeText(getBaseContext(), "You need more to call train: "+ remainigPhotos, Toast.LENGTH_SHORT).show();
            Log.d("Piopr", "Test do cholery");
            return;
        }else if(TrainHelper.isTrained(getBaseContext())) {
            Toast.makeText(getBaseContext(), "Already trained", Toast.LENGTH_SHORT).show();
            return;
        }

        Toast.makeText(getBaseContext(), "Start train: ", Toast.LENGTH_SHORT).show();
        new AsyncTask<Void, Void, Void>() {

            @Override
            protected Void doInBackground(Void... voids) {
                try{
                    if(!TrainHelper.isTrained(getBaseContext())) {
                        TrainHelper.train(getBaseContext());
                    }
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                try {
                    Toast.makeText(getBaseContext(), "Reseting after train - Sucess : "+ TrainHelper.isTrained(getBaseContext()), Toast.LENGTH_SHORT).show();
                    finish();
                }catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }
        }.execute();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        absoluteFaceSize = (int) (width * 0.32f);
    }

    @Override
    public void onCameraViewStopped() {

    }

    /***
     * wykonanie zdjęcia
     * TODO: zmiana id odpowiednio do użytkownika
     * @param rgbaMat - obraz przechwycony z cameraactivity
     *
     * Po wykonaniu zdjęcia zmiana takePhoto na false
     */
    private void capturePhoto(Mat rgbaMat) {
        try {
            TrainHelper.takePhoto(getBaseContext(), 1, TrainHelper.qtdPhotos(getBaseContext()) + 1, rgbaMat.clone(), faceDetector);
        }catch (Exception e) {
            e.printStackTrace();
        }
        takePhoto = false;
    }

    /***
     *
     * @param dadosFace - zakres w którym znajduje się twarz w momencie przechwycenia zdjęcia
     * @param grayMat - przechwycone zdjęcie w grayscale
     * @param rgbaMat - przechwycone zdjęcie w kolorze + kanał alfa
     */
    private void recognize(opencv_core.Rect dadosFace, Mat grayMat, Mat rgbaMat) {
        Mat detectedFace = new Mat(grayMat, dadosFace);
        resize(detectedFace, detectedFace, new Size(TrainHelper.IMG_SIZE,TrainHelper.IMG_SIZE));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);
        faceRecognizer.predict(detectedFace, label, reliability);
        int prediction = label.get(0);
        double acceptanceLevel = reliability.get(0);

        String name;
        if (prediction == -1 || acceptanceLevel >= ACCEPT_LEVEL) {
            name = getString(R.string.unknown);
        } else {
            name = nomes[prediction] + " - " + cvRound(acceptanceLevel) + " label: " + prediction;
        }
        int x = Math.max(dadosFace.tl().x() - 10, 0);
        int y = Math.max(dadosFace.tl().y() - 10, 0);
        putText(rgbaMat, name, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0,255,0,0));
    }

    void showDetectedFace(RectVector faces, Mat rgbaMat) {
        int x = faces.get(0).x();
        int y = faces.get(0).y();
        int w = faces.get(0).width();
        int h = faces.get(0).height();

        rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);
    }

    void noTrainedLabel(opencv_core.Rect face, Mat rgbaMat) {
        int x = Math.max(face.tl().x() - 10, 0);
        int y = Math.max(face.tl().y() - 10, 0);
        putText(rgbaMat, "No trained or train unavailable", new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0,255,0,0));
    }

    @Override
    public Mat onCameraFrame(Mat rgbaMat) {
        if (faceDetector != null) {
            Mat greyMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
            cvtColor(rgbaMat, greyMat, CV_BGR2GRAY);
            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));

            if (faces.size() == 1) {
                showDetectedFace(faces, rgbaMat);
                if(takePhoto) {
                    capturePhoto(rgbaMat);
                    alertRemainingPhotos();
                }
                if(trained) {
                    recognize(faces.get(0), greyMat, rgbaMat);
                }else{
                    noTrainedLabel(faces.get(0), rgbaMat);
                }
            }
            greyMat.release();
        }
        return rgbaMat;
    }

    void alertRemainingPhotos() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                int remainigPhotos = TrainHelper.PHOTOS_TRAIN_QTY - TrainHelper.qtdPhotos(getBaseContext());
                Toast.makeText(getBaseContext(), "You need more to call train: "+ remainigPhotos, Toast.LENGTH_SHORT).show();
            }
        });
    }
}