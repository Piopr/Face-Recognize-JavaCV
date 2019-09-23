package com.example.piotr.androidrecognizer;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
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

import static com.example.piotr.androidrecognizer.TrainHelper.ACCEPT_LEVEL_LBPH;
import static com.example.piotr.androidrecognizer.TrainHelper.CURRENT_IDUSER;
import static com.example.piotr.androidrecognizer.TrainHelper.FISHER_EXISTS;
import static com.example.piotr.androidrecognizer.TrainHelper.IS_TRAINED;
import static com.example.piotr.androidrecognizer.TrainHelper.VERIFIED;
import static com.example.piotr.androidrecognizer.TrainHelper.checkFisherExists;

import static com.example.piotr.androidrecognizer.TrainHelper.isTrained;
import static com.example.piotr.androidrecognizer.TrainHelper.makeMeanFaces;
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
 * Struktura plikow:
 * mnt/sdcard/trainfolder/
 *      ---/1user/
 *              ---/default/
 *                  ---zdjecie.jpg
 *              ---/visualization/
 *
 *              --person.id.numberofphoto.jpg
 *              --person.1.2.jpg
 *              --person.1.3.jpg
 *              --...
 *      ---/2user/
 *              ---/default/
 *              ---/visualization/
 *              --person.2.1.jpg
 *              --person.2.2.jpg
 *              --...
 *      ---eigenFacesClassifier.yml
 *      ---fisherFacesClassifier.yml
 *      ---lbphClassifier.yml
 *
 * OPIS:
 * Wszystko zapisywane jest w pamieci wewnetrznej telefonu.
 * /trainfolder/ - glówny katalog aplikacji
 * /1user/, /2user/ - katalogi uzytkownikow *
 * /
 */

public class OpenCvRecognizeActivity extends Activity implements CvCameraPreview.CvCameraViewListener {
    public static final String TAG = "Piopr";
    /**
     * zmienna CascadeClassifiera, czyli wzorca do detekcji twarzy
     */
    private CascadeClassifier faceDetector;
    /**
     *  Nazwy stworzonych uzytkownikow.
     *  Wykorzystywane do wyswietlania komunikatow
     *  {"user1", "user2"}, element 0 jest pusty
     */
    private String[] usersNamesArray;// = TrainHelper.getUserNames();

    /**
     * Zawiera id uzytkownikow
     */
    private Integer[] usersIdArray;

    /**
     * minimalny rozmiar twarzy na activity kamery. Minimalna wartość, to 1/5 wysokosci lub 160;
     */
    private int absoluteFaceSize = 0;
    /**
     * obiekt do sterowania kamerą
     */
    private CvCameraPreview cameraView;
    /**
     * Kontrola momentu robienia zdjecia.
     * gdy zdjęcie jest wykonywane - true
     * gdy nie jest lub zakońcono zmieniana na - false
     */
    boolean takePhoto;

    /**
     * Obiekty klas do algorytmow rozpoznawania twarzy
     */
    opencv_face.FaceRecognizer faceEigen = opencv_face.EigenFaceRecognizer.create();
    opencv_face.FaceRecognizer faceFisher = opencv_face.FisherFaceRecognizer.create();
    opencv_face.FaceRecognizer faceLBPH = opencv_face.LBPHFaceRecognizer.create();


    /**
     * jesli już nauczono twarzy (istnieja przynajmniej 2 pliki .yml) zmienia się na true i można zacząć rozpoznawanie twarzy
     */
    boolean trained;

    /**
     * Zmienna, a w zasadzie flaga informujaca, czy aktualny uzytkownik zgadza sie z tym rozpoznanym.
     * Poprawnosc widziana w kolorze jakim jest kwadrat wyswietlajacy twarz.
     */
    private boolean verified = false;

    /**     *
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
            String[] PERMISSIONS = {android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
            if (!hasPermissions(this, PERMISSIONS)) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, 1);
            }
        }
        /*
        wczytanie z activity zmiennej obsługującej podgląd kamery (wykrywanie twarzy itd.)
         */
        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        /*
        Zaladowanie nazw i id userow (stworzonych folderow)
         */
        usersNamesArray = TrainHelper.getUserNames();
        Log.d("Piopr", "Dlugosc listy userow: " + usersNamesArray.length);

        VERIFIED = false;//Poczatkowa inicjalizacja na false

        /*
        wykonywanie operacji w tle,
        wczytywanie istniejacych algorytmow
         */
        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {

                try {
                    /***
                     * wczytanie wzorca do detekcji twarzy
                     */
                    faceDetector = TrainHelper.loadClassifierCascade(OpenCvRecognizeActivity.this, R.raw.frontalface);
                    //Wczytanie z plikow (jesli istnieja), plikow z wytrenowanymi algorytmami.
                    //Wymagane 3 lub 2 (w przypadku, gdy tylko 1 zestaw zdjec nie mozna wytrenowac FisherFaces)
                    if (TrainHelper.isTrained(getBaseContext())) {
                        File folder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
                        File f = new File(folder, TrainHelper.EIGEN_FACES_CLASSIFIER);
                        faceEigen.read(f.getAbsolutePath());
                        //Sprawdzenie, czy jest wytrenowany
                        TrainHelper.FISHER_EXISTS = TrainHelper.checkFisherExists();
                        if (FISHER_EXISTS) {

                            f = new File(folder, TrainHelper.FISHER_FACES_CLASSIFIER);
                            faceFisher.read(f.getAbsolutePath());

                        }

                        f = new File(folder, TrainHelper.LBPH_CLASSIFIER);
                        faceLBPH.read(f.getAbsolutePath());


                    } else {
                        Toast.makeText(getBaseContext(), "Algorytmy niewytrenowane.", Toast.LENGTH_SHORT).show();
                    }
                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            /***
             *
             * Po zakończeniu sprawdzania nauczenia algorytmu
             * Obsługa przycisków i włączanie ich widocznośći.
             *
             */
            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);

                if(isTrained(getBaseContext())){
                    trained = true;
                    IS_TRAINED = true;
                    Log.d("Piopr", "Is Trained");
                }
                //Wykonanie zdjecia
                findViewById(R.id.btPhoto).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        takePhoto = true;
                    }
                });
                findViewById(R.id.btPhoto).setEnabled(true);
                //Trening algorytmu
                findViewById(R.id.btTrain).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            train();
                        } catch (Exception e) {
                            Log.d("Piopr", e.getLocalizedMessage(), e);
                        }


                    }
                });
                findViewById(R.id.btTrain).setEnabled(true);

                /**
                 Resetowanie algorytmow (bez usuwania zdjec).
                 Po zakończeniu powrót do poprzedniego acitvity
                 */
                findViewById(R.id.btReset).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            TrainHelper.reset(getBaseContext());
                            Toast.makeText(getBaseContext(), "Zresetowano algorytm.", Toast.LENGTH_SHORT).show();
                            finish();
                        } catch (Exception e) {
                            Log.d(TAG, e.getLocalizedMessage(), e);
                        }
                    }
                });
                findViewById(R.id.btReset).setEnabled(true);

                //Wykrycie twarzy na zdjęciach w folderze użytkownika.
                //Zamienia oryginalne na samą twarz, 160x160px grayscale
                findViewById(R.id.btDetect).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            TrainHelper.detectFaceFromPhotos(getBaseContext(), faceDetector, TrainHelper.CURRENT_FOLDER);
                        } catch (Exception e) {
                            Log.d("Piopr", e.getLocalizedMessage(), e);
                        }
                    }
                });
                findViewById(R.id.btDetect).setEnabled(true);
                //Rozpoznanie ze zdjecia znajdujacego sie w ./default
                findViewById(R.id.btZeZdjecia).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            TrainHelper.recognizeFromPhoto(getBaseContext(), TrainHelper.CURRENT_FOLDER);

                        } catch (Exception e) {
                            Log.d("Piopr", e.getLocalizedMessage(), e);
                        }
                    }
                });
                findViewById(R.id.btZeZdjecia).setEnabled(true);
                //Tworzenie wizualizacji z algortmow
                findViewById(R.id.btMean).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            //makeMeanFaces(getBaseContext());
                            trained = false;
                            makeMeanFaces(getBaseContext());
                            if (isTrained(getBaseContext())) {
                                trained = true;
                            }
                        } catch (Exception e) {
                            Log.d("Piopr", e.getLocalizedMessage(), e);
                        }
                    }
                });
                findViewById(R.id.btMean).setEnabled(true);
                if (isTrained(getBaseContext())) {

                    if (!checkFisherExists()) {
                        Toast.makeText(getBaseContext(), "Aby wytrenować algorytm Fisherfaces potrzeba przynajmniej dwóch zestawow zdjec", Toast.LENGTH_SHORT).show();
                    }
                }

                /*
                  Zmiana nazwy zdjec w folderze uzytkownika
                  na zgodny z patternem "person.id.nr_zdjecia.jpg
                 */
                findViewById(R.id.btPrzygotuj).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        TrainHelper.renamePhotos();
                        //TrainHelper.listPhotos();
                    }
                });
                findViewById(R.id.btPrzygotuj).setEnabled(true);

                /*
                Test porownania wszystkich uzytkownikow.
                 */
                findViewById(R.id.btTesting).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            TrainHelper.predictTest(getBaseContext());
                        } catch (Exception e) {
                            Log.d("Piopr", e.getLocalizedMessage(), e);
                        }
                        //TrainHelper.listPhotos();
                    }
                });
                findViewById(R.id.btTesting).setEnabled(true);

                PomocRecognizeDialog dialog = new PomocRecognizeDialog();
                findViewById(R.id.btHelp).setOnClickListener(view ->
                       dialog.show(getFragmentManager(), "Piopr") );


                findViewById(R.id.loading).setVisibility(View.GONE);
            }
        }.execute();


    }

    /***
     *      Metoda do treningu algorytmu
     *  1. Sprawdzenie, czy algorytmy juz wytrenowane
     *      Jesli tak, komunikat. Jesli nie - trening.
     */
    @SuppressLint("StaticFieldLeak")
    void train() {
        //int remainigPhotos = TrainHelper.PHOTOS_TRAIN_QTY - TrainHelper.qtdPhotos(getBaseContext());
        if (TrainHelper.isTrained(getBaseContext())) {
            Toast.makeText(getBaseContext(), "Algorytm juz wytrenowany.", Toast.LENGTH_SHORT).show();
            return;
        }

        Toast.makeText(getBaseContext(), "Rozpoczęto trening. \n Czekaj... ", Toast.LENGTH_SHORT).show();
        new AsyncTask<Void, Void, Void>() {

            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    if (!TrainHelper.isTrained(getBaseContext())) {
                        TrainHelper.train(getBaseContext());
                    }
                } catch (Exception e) {
                    Log.d("Piopr", e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                try {
                    Toast.makeText(getBaseContext(), "Koniec treningu. Status: " + TrainHelper.isTrained(getBaseContext()), Toast.LENGTH_SHORT).show();
                    finish();
                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }
        }.execute();
    }

    /*
    Moment startu kamery.
    Obliczenie maksymalnego rozmiaru wykrywanej twarzy na zdjeciu.
     */
    @Override
    public void onCameraViewStarted(int width, int height) {
        if ((int) (width * 0.2f) > 160) {
            absoluteFaceSize = (int) (width * 0.2f);
        } else {
            absoluteFaceSize = 160;
        }
    }

    @Override
    public void onCameraViewStopped() {

    }

    /***
     *          Wykonanie zdjęcia
     *   Zapisanie go jako:</br> \n
     *   -160x160px</br>\n
     *   -grayscale</br>\n
     *   -zgodnie z patternem person.id.nr_zdjecia.jpg</br>\n
     *   Po wykonaniu zdjęcia zmiana takePhoto na false
     * @param rgbaMat - obraz przechwycony z cameraactivity
     */
    private void capturePhoto(Mat rgbaMat) {
        try {
            TrainHelper.takePhoto(getBaseContext(), TrainHelper.CURRENT_IDUSER, TrainHelper.qtdPhotosNew() + 1, rgbaMat.clone(), faceDetector, TrainHelper.CURRENT_FOLDER);
        } catch (Exception e) {
            e.printStackTrace();
        }
        takePhoto = false;
    }

    /***
     *  Rozpoznanie twarzy w activity, wyswietlenie prostokata gdzie znajduje sie twarz,<br>
     *  Wyswietlenie odpowienich komunikatow jesli rozpoznano lub nie.<br>
     *      Jesli tak:<br>
     *          kolor czerwony, jesli istnieje w bazie, ale nie jest aktualnym<br>
     *          kolor zielony jesli tak i aktualny uzytkownik<br>
     *      Jesli nie komunikat o nierozpoznaniu<br>
     *   Metoda sprawdza dodatkowo, czy wytrenowano Fisherface w przypadku tylko jednego zestawu zdjęć.
     * @param rectFace - zakres w którym znajduje się twarz w momencie przechwycenia zdjęcia
     * @param grayMat - przechwycone zdjęcie w grayscale
     * @param rgbaMat - przechwycone zdjęcie w kolorze + kanał alfa
     */
    private void recognize(opencv_core.Rect rectFace, Mat grayMat, Mat rgbaMat) {

        Mat detectedFace = new Mat(grayMat, rectFace);
        resize(detectedFace, detectedFace, new Size(TrainHelper.IMG_SIZE, TrainHelper.IMG_SIZE));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);
        faceEigen.predict(detectedFace, label, reliability);
        int prediction = label.get(0);


        //sprawdzanie zawartosci zmiennej label
        //Log.d("Piopr", "label.get(0): " + label.get(0));
        double acceptanceLevel = reliability.get(0);

        String name;
        if (prediction == -1 || acceptanceLevel >= ACCEPT_LEVEL) {
            name = getString(R.string.unknown);
        } else {
            name = "Witaj " + usersNamesArray[prediction] + "! - " + cvRound(acceptanceLevel) + " id: " + prediction;
        }
        int x = Math.max(rectFace.tl().x() - 10, 0);
        int y = Math.max(rectFace.tl().y() - 10, 0);

        if (CURRENT_IDUSER == prediction && acceptanceLevel < ACCEPT_LEVEL) {
            putText(rgbaMat, name, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0, 255, 0, 0));

        } else {
            putText(rgbaMat, name, new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255, 0, 0, 0));
        }


        if (checkFisherExists()) {
            //algorytm fisher
            faceFisher.predict(detectedFace, label, reliability);

            prediction = label.get(0);
            //sprawdzanie zawartosci zmiennej label
            //Log.d("Piopr", "label.get(0): " + label.get(0));
            acceptanceLevel = reliability.get(0);

            if (prediction == -1 || acceptanceLevel >= ACCEPT_LEVEL) {
                name = getString(R.string.unknown);
            } else {
                name = "Witaj " + usersNamesArray[prediction] + "! - " + cvRound(acceptanceLevel) + " id: " + prediction;
            }
            if (CURRENT_IDUSER == prediction && acceptanceLevel <= ACCEPT_LEVEL) {
                putText(rgbaMat, name, new Point(x, y - 20), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0, 255, 0, 0));
            } else {
                putText(rgbaMat, name, new Point(x, y - 20), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255, 0, 0, 0));
            }
        } else {
            putText(rgbaMat, "Alg. Fisher niewytrenowany.", new Point(x, y - 20), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255, 0, 0, 0));

        }

        //algorytm LBPH

        faceLBPH.predict(detectedFace, label, reliability);

        prediction = label.get(0);
        //sprawdzanie zawartosci zmiennej label
        //Log.d("Piopr", "label.get(0): " + label.get(0));
        acceptanceLevel = reliability.get(0);

        if (prediction == -1 || acceptanceLevel >= ACCEPT_LEVEL_LBPH) {
            name = getString(R.string.unknown);
            verified = false;
        } else {
            name = "Witaj " + usersNamesArray[prediction] + "! - " + cvRound(acceptanceLevel) + " id: " + prediction;
        }
        if (CURRENT_IDUSER == prediction && acceptanceLevel < ACCEPT_LEVEL_LBPH) {
            putText(rgbaMat, name, new Point(x, y - 40), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(0, 255, 0, 0));
            verified = true;
        } else {
            putText(rgbaMat, name, new Point(x, y - 40), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255, 0, 0, 0));
            verified = false;
        }
    }

    /***
     * Wyswietlanie prostokata w odpowiednim kolorze na ekranie w miejscu, gdzie jest twarz.
     * @param faces - wspolrzedne, na ktorych znajduje sie wykryta twarz.
     * @param rgbaMat - aktualna klatka.
     */
    void showDetectedFace(RectVector faces, Mat rgbaMat) {
        int x = faces.get(0).x();
        int y = faces.get(0).y();
        int w = faces.get(0).width();
        int h = faces.get(0).height();
        if(verified) {
            rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), new opencv_core.Scalar(0,255,0,0), 2, LINE_8, 0);
        } else {
            rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), new opencv_core.Scalar(255,0,0,0), 2, LINE_8, 0);
        }
    }

    void noTrainedLabel(opencv_core.Rect face, Mat rgbaMat) {
        int x = Math.max(face.tl().x() - 10, 0);
        int y = Math.max(face.tl().y() - 10, 0);
        putText(rgbaMat, "Algorytm niewytrenowany", new Point(x, y-20), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255,0,0,0));
        putText(rgbaMat, "lub niezaladowany.", new Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new opencv_core.Scalar(255,0,0,0));
    }

    /***
     * Moment wyswietlania klatki.<br>
     * Detekcja twarzy na na klatce, narysowanie kwadratu, sprawdzenie, czy rozpoznano.
     *
     * @param rgbaMat - przechwycona klatka
     * @return zmieniona klatka
     */
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

                if (takePhoto) {
                    capturePhoto(rgbaMat);
                    alertQtyPhotos();
                }
                showDetectedFace(faces, rgbaMat);
                if (IS_TRAINED && trained) {
                    recognize(faces.get(0), greyMat, rgbaMat);
                } else {
                    noTrainedLabel(faces.get(0), rgbaMat);
                }
            }

            greyMat.release();
        }
        if (VERIFIED) {
            runOnUiThread(new Runnable() {

                @Override
                public void run() {

                    findViewById(R.id.verified).setVisibility(View.VISIBLE);
                }
            });

        }
        return rgbaMat;
    }

    /***
     * Informuje o ilosci zrobionych zdjec.
     */
    void alertQtyPhotos() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                //int remainigPhotos = TrainHelper.qtdPhotos(getBaseContext());
                Toast.makeText(getBaseContext(), "Zdjecie nr: " + TrainHelper.qtdPhotosNew(), Toast.LENGTH_SHORT).show();
            }
        });
    }

}