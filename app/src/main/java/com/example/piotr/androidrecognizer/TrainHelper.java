/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.example.piotr.androidrecognizer;

import android.content.Context;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.IntBufferIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_tracking;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;


import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.GEMM_2_T;
import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.PCACompute;
import static org.bytedeco.javacpp.opencv_core.PCACompute2;
import static org.bytedeco.javacpp.opencv_core.add;
import static org.bytedeco.javacpp.opencv_core.cvRound;
import static org.bytedeco.javacpp.opencv_core.eigen;
import static org.bytedeco.javacpp.opencv_core.gemm;
import static org.bytedeco.javacpp.opencv_core.noArray;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_core.read;
import static org.bytedeco.javacpp.opencv_core.shiftLeft;
import static org.bytedeco.javacpp.opencv_core.subtract;
import static org.bytedeco.javacpp.opencv_core.transpose;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_GRAY2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_GRAY2BGRA;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_GRAY2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.threshold;

/**
 * @author djalmaafilho
 */
public class TrainHelper {

    /**
     * tag logowania
     */
    public static final String TAG = "TrainHelper";
    /**
     * nazwa folderu przechowujacego zdjecia oraz pliki .yml
     */
    public static final String TRAIN_FOLDER = "train_folder";
    /**
     * rozmiar, do którego skalowana będzie twarz (160px x 160px)
     */
    public static final int IMG_SIZE = 160;
    /**
     * nazwy klasyfikatorów dla każdego z algorytmów
     */
    public static final String EIGEN_FACES_CLASSIFIER = "eigenFacesClassifier.yml";
    public static final String FISHER_FACES_CLASSIFIER = "fisherFacesClassifier.yml";
    public static final String LBPH_CLASSIFIER = "lbphClassifier.yml";
    /**
     * patern dla zapisywanyh zdjęć zrobionych aparatem
     */
    public static final String FILE_NAME_PATTERN = "person.%d.%d.jpg";
    /**
     * ilosc zdjec wymaganch do trenowania
     */
    public static final int PHOTOS_TRAIN_QTY = 25;
    /**
     * współczynnik po którym aplikacja stwierdza, że porównywana twarz jest znajoma
     * poniżej 4000 - rozpoznanie
     * poniżej 4000 - unknown
     */
    public static final double ACCEPT_LEVEL = 4000.0D;
    public static final double ACCEPT_LEVEL_LBPH = 60.0D;

    public static String CURRENT_USER;
    public static int CURRENT_IDUSER;
    public static String CURRENT_FOLDER;
    public static boolean FISHER_EXISTS;
    public static boolean VERIFIED = false;


    public static boolean IS_TRAINED;

    /**
     * obsługa przycisku reset (usuniecie plikow z algorytmem rozpoznawania)
     *
     * @param context
     */
    public static void reset(Context context) throws Exception {
        //File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER);

        Log.d("Piopr " + TAG, photosFolder.toString());
        if (photosFolder.exists()) {

            FilenameFilter imageFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png") || name.endsWith(".yml");
                }
            };

            FilenameFilter trainFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".yml");
                }
            };

            File[] files = photosFolder.listFiles(trainFilter);

            for (File file : files) {
                file.delete();
            }
        }
    }

    /**
     * sprawdzenie, czy już trenowano algorytm
     *
     * @param context
     * @return
     */
    public static boolean isTrained(Context context) {
        try {
            //File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
            File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER);
            if (photosFolder.exists()) {

                FilenameFilter imageFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
                    }
                };

                FilenameFilter trainFilter = new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.endsWith(".yml");
                    }
                };


                File[] photos = photosFolder.listFiles(imageFilter);
                File[] train = photosFolder.listFiles(trainFilter);

                return train != null && train.length >= 2;
            } else {
                return false;
            }

        } catch (Exception e) {
            Log.d(TAG, e.getLocalizedMessage(), e);
        }
        return false;
    }

    /**
     * Służy oliczenie ilość plików zdjęciowych znajdujących się w folderrze treningu
     *
     * @param context - aktualny widok
     * @return - ilosc zdjęć
     */
    public static int qtdPhotos(Context context) {
        //File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER);
        if (photosFolder.exists()) {
            FilenameFilter imageFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
                }
            };

            File[] files = photosFolder.listFiles(imageFilter);
            return files != null ? files.length : 0;
        }
        return 0;
    }

    public static int qtdPhotosNew() {
        //File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER);
        if (photosFolder.exists()) {
            FilenameFilter imageFilter = new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
                }
            };

            File[] files = photosFolder.listFiles(imageFilter);
            return files != null ? files.length : 0;
        }
        return 0;
    }

    /**
     * Klasa służąca do treningu.
     * Na początku sprawdza, czy istnieje folder ze zdjęciami do trenowania. Jeśli nie - zwraca false.</br>
     * Nastepnie w zmiennej Files[] files tworzy listę zdjęć.</br>
     * Tworzy wskaźnik na zdjęcia typu MatVector photos o rozmiarze równym ilości plików do trenowania.</br>
     * Tworzy zbiór etykiet labels (Mat labels). Ilość wierszy: ilość plików. Ilość kolumn: 1, typu 32-bitowych shortów z jednym kanałem</br>
     * rotulosBuffer nie wiem do czego jest potrzebny.</br>
     * Przechodzi po liśie plików files</br>
     * Zapisuje w timczasowej Mat photo aktualne zdjęcie w grayscale</br>
     * classe to numer identyfikacyjny wyciągany z nazwy zdjęcia</br>
     * przeskalowuje obraz (wejsciowym jest kwadratowa twarz o nieznanych rozmiarach) na rozmiar IMG_SIZE px x IMG_SIZE px</br>
     * taki obraz umieszczany jest w liście MatVector photos</br></br>
     * <p>
     * Tworzone są obiekty klas dla algorytmów rozpoznawania np:</br> <b>FaceRecognizer eigenfaces = opencv_face.EigenFaceRecognizer.create();</b>
     * <p>
     * następnie odbywa się trening: <b>eigenfaces.train(photos, labels);</b>.
     * photos to MatVector ze zdjęciami twarzy 160px x 160px
     * labels to Mat rozmiarem odpowiadajcy ilosci trenowanych zdjęć
     * <p>
     * <p>
     * <p>
     * wynik treningu zapisywany jest metodą <b>eigenfaces.save(f.getAbsolutePath());</b> to pliku zdefoniowanego stałą *_FACES_CLASIFIER
     *
     * @param context
     * @return zwraca bool. Prawda, gdy istnieją obrazy do trenowania i wykonano trening, false, gdy zdjęcia nie istnieją
     */
    public static boolean train(Context context) throws Exception {

        //File photosFolder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER);
        if (!photosFolder.exists()) return false;

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };


        File[] listOfUserFolders = photosFolder.listFiles(new FileFilter() {
            @Override
            public boolean accept(File file) {

                return file.isDirectory();
            }
        });


        List<File> photosList = new ArrayList<File>();
        for (File f : listOfUserFolders) {
            List<File> tmpPhotosList = Arrays.asList(f.listFiles(imageFilter));
            photosList.addAll(tmpPhotosList);
        }

        File[] files = new File[photosList.size()];
        int counter = 0;
        for (File f : photosList) {
            files[counter] = f;
            counter++;
        }

        for (File f : files) {
            Log.d("Piopr", f.getAbsolutePath());
        }

        //File[] files = photosFolder.listFiles(imageFilter);
        MatVector photos = new MatVector(files.length);
        Mat labels = new Mat(files.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = labels.createBuffer();
        counter = 0;
        for (File image : files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);
            //Log.d("Piopr", "Numer id: " + classe);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);
            rotulosBuffer.put(counter, classe);
            counter++;
        }


        //labels.create

        //IntRawIndexer idBuffer = labels.createIndexer();
        //IntBufferIndexer idBuffer = labels.createIndexer();
        IntIndexer idBuffer = labels.createIndexer();


        //labels.


//        FaceRecognizer eigenfaces = createEigenFaceRecognizer();
//        FaceRecognizer fisherfaces = createFisherFaceRecognizer();
//        FaceRecognizer lbph = createLBPHFaceRecognizer(2,9,9,9,1);

        FaceRecognizer eigenfaces = opencv_face.EigenFaceRecognizer.create();
        FaceRecognizer fisherfaces = opencv_face.FisherFaceRecognizer.create();
        FaceRecognizer lbph = opencv_face.LBPHFaceRecognizer.create();


        eigenfaces.train(photos, labels);
        File f = new File(photosFolder, EIGEN_FACES_CLASSIFIER);
        f.createNewFile();
        eigenfaces.save(f.getAbsolutePath());



        if (checkCouplePersonsExists(idBuffer)) {
            fisherfaces.train(photos, labels);
            f = new File(photosFolder, FISHER_FACES_CLASSIFIER);
            f.createNewFile();
            fisherfaces.save(f.getAbsolutePath());
        }



        lbph.train(photos, labels);

        Log.d("Piopr", "th: " + lbph.getThreshold());
        f = new File(photosFolder, LBPH_CLASSIFIER);
        f.createNewFile();
        lbph.save(f.getAbsolutePath());

        return true;
    }

    public static void makeMeanFaces(Context context) throws Exception {
        File visPath = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + CURRENT_FOLDER + "/" + "visualizations/");
        if(!visPath.exists()){
            visPath.mkdir();
        }

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        File userphotosFolder = new File("/mnt/sdcard/"+TRAIN_FOLDER, CURRENT_FOLDER);
        File[] files = userphotosFolder.listFiles(imageFilter);
        if(files.length<1){
            Log.d("Piopr", "Brak zdjec!");
            return;
        }

        MatVector photos = new opencv_core.MatVector(files.length+1);
        Mat labels = new opencv_core.Mat(files.length+1, 1, CV_32SC1);
        IntBuffer rotulosBuffer = labels.createBuffer();
        int counter = 0;

        for (File image : files) {
            Mat photo = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(image.getName().split("\\.")[1]);
            Log.d("Piopr", "Numer id: " + classe);

            resize(photo, photo, new opencv_core.Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);

            rotulosBuffer.put(counter, classe);
            counter++;
        }
        photos.put(counter, imread(files[0].getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE));
        rotulosBuffer.put(counter, Integer.parseInt(files[0].getName().split("\\.")[1])+1);



        opencv_face.FaceRecognizer eigenfaces = opencv_face.EigenFaceRecognizer.create();
        opencv_face.FaceRecognizer fisherfaces = opencv_face.FisherFaceRecognizer.create();
        opencv_face.FaceRecognizer lbph = opencv_face.LBPHFaceRecognizer.create();

        eigenfaces.train(photos, labels);

        Mat eigenVectors = ((opencv_face.EigenFaceRecognizer) eigenfaces).getEigenVectors();
        Mat eigenMean = ((opencv_face.EigenFaceRecognizer) eigenfaces).getMean();

        for (int i = 0; i < files.length; i++) {
            Mat wektorek = eigenVectors.col(i);
            transpose(wektorek, wektorek);
            Mat output = wektorek.reshape(1, 160);
            normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
            imwrite(visPath + "/eigenVec" + i + ".jpg", output);
        }

        eigenMean = eigenMean.reshape(1,IMG_SIZE);
        imwrite(visPath + "/meanEigen.jpg", eigenMean);



        fisherfaces.train(photos, labels);

        Mat fisherVectors = ((opencv_face.FisherFaceRecognizer) fisherfaces).getEigenVectors();
        Mat fisherMean = ((opencv_face.FisherFaceRecognizer) fisherfaces).getMean();

        Log.d("Piopr", "Fisher cols: " + fisherVectors.cols());

//        for (int i = 0; i < files.length; i++) {
//            Mat wektorek = fisherVectors.col(i);
//            transpose(wektorek, wektorek);
//            Mat output = wektorek.reshape(1, 160);
//            normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
//            imwrite(visPath + "/fisherVec" + i + ".jpg", output);

        fisherVectors = fisherVectors.reshape(1,160);
        normalize(fisherVectors, fisherVectors, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
        imwrite(visPath + "/fisherVec.jpg", fisherVectors);

        fisherMean = eigenMean.reshape(1,IMG_SIZE);
        imwrite(visPath + "/meanfisher.jpg", fisherMean);


        //Mat

        lbph.train(photos,labels);


        MatVector histo = ((opencv_face.LBPHFaceRecognizer) lbph).getHistograms();
        Log.d("Piopr", "Histo rows: " + histo.get(0).rows());
        Log.d("Piopr", "Histo cols: " + histo.get(0).cols());
        Log.d("Piopr", "Histo total: " + histo.get(0).total());
        Log.d("Piopr", "Histo size: " + histo.size());
        Mat output = histo.get(10);
        normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
        imwrite(visPath+"/histo.jpg", output.reshape(1,128));
        int radius = ((opencv_face.LBPHFaceRecognizer) lbph).getRadius();
        int neighbours = ((opencv_face.LBPHFaceRecognizer) lbph).getNeighbors();
        int gridX = ((opencv_face.LBPHFaceRecognizer) lbph).getGridX();
        int gridY = ((opencv_face.LBPHFaceRecognizer) lbph).getGridY();
        double th = lbph.getThreshold();
        double th2 = th - 1.0d;
        Log.d("Piopr", "th2: " + th2);
        Log.d("Piopr", "th2: " + lbph.getThreshold());





        Mat lbpout = new Mat();

        threshold(photos.get(10), lbpout, 128.797, 255, 0);
        normalize(lbpout, lbpout, 0, 255, NORM_MINMAX, CV_8UC1, noArray());
        imwrite(visPath+"/lbpout.jpg", lbpout);



        Log.d("Piopr", "radius: " + radius + ", neighbours: " + neighbours+
                "\n gridX: "+ gridX + " gridY: " + gridY + " th: " + th);
    }

    /**
     * zapis zdjęcia
     * <p>
     * Na początku kontrola, czy istnieje folder. Jeśli nie - tworzy go
     * Stworzenie kopii zdjęcia w zmiennej greyMat klasy Mat. Przekształcenie na greyscale
     * Stworzenie wektora o 4 współrzędnych (współrzędne tworzonego kwadratu podczas detekcji twarzy), zmienna detectedFaces.
     * <p>
     * zdjęcie zapisuje się w formacie: person.personId.photonumber.jpg
     * <p>
     * użycie metody detectMultiScale() na obiekcie faceDetector:
     * detectMultiScale(greyMat, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
     * greyMat - zdjęcie z którygo chcemy wykryć twarz
     * detectedFaces - obiekt, do którego zapisujemy współrzędne, w których została wykryta twarz na zdjęciu
     * 1.1 - współczynnik, który określa o ile wielkośc obrazu zostanie zmniejszona
     * 1 - Parametr określający, ilu sąsiadów każdy kandydujący prostokąt powinien zachować.
     * 0 - używane w starszych wersjach cascadeClassifier
     * 150, 150 - minimalny rozmiar w pikselach fragmentu zdjęcia, na którym znajduje się twarz
     * 500, 500 - maksymalny rozmiar na zdjęciu, na którym może znaleźć się twarz
     * <p>
     * wykonuje się pętla przechodząca po wszystkich wykrytych tawrzach (zwykle, i poprawnie po jednej),
     * w tej pętli wszystkie wykryte twarze nadpisywane są do wcześniej stworzonego pliku .jpg w skali szarości i rozmiarze określonym
     * w zmiennej IMG_SIZE
     *
     * @param context      - aktualny widok
     * @param personId     - id osoby - dodawane do nazwy pliku
     * @param photoNumber  - numer zrobionego zdjęcia, dodawane do nazwy pliku
     * @param rgbaMat      - oryginalny obrazek, przechwycony z cameraactivity
     * @param faceDetector - obiekt CascadeClassifier z pliku frontalface.xml
     */
    public static void takePhoto(Context context, int personId, int photoNumber, Mat rgbaMat, opencv_objdetect.CascadeClassifier faceDetector, String personDirName) throws Exception {
        //File folder = new File(context.getFilesDir(), TRAIN_FOLDER);

        File folder = new File("/mnt/sdcard/", TRAIN_FOLDER + "/" + personDirName);
        Log.d("Piopr", folder.toString());
        if (folder.exists() && !folder.isDirectory())
            folder.delete();
        if (!folder.exists())
            folder.mkdirs();

        Mat greyMat = new Mat(rgbaMat.rows(), rgbaMat.cols());

        cvtColor(rgbaMat, greyMat, CV_BGR2GRAY);
        opencv_core.RectVector detectedFaces = new opencv_core.RectVector();
        faceDetector.detectMultiScale(greyMat, detectedFaces, 1.1, 1, 0, new Size(160, 160), new Size(500, 500));
        for (int i = 0; i < detectedFaces.size(); i++) {
            //TODO: zmiana funkcji, by zapisywała zdjęcia w odpowiednim folderze, obsłużenie parametru personDirName, poprawa metody qtdPhotos

            opencv_core.Rect rectFace = detectedFaces.get(0);
            Log.d("Piopr", "rectFace :  " + rectFace.get());
            rectangle(rgbaMat, rectFace, new opencv_core.Scalar(0, 0, 255, 0));


            Mat capturedFace = new Mat(greyMat, rectFace);
            resize(capturedFace, capturedFace, new Size(IMG_SIZE, IMG_SIZE));

            File f = new File(folder, String.format(FILE_NAME_PATTERN, personId, photoNumber));
            f.createNewFile();
            imwrite(f.getAbsolutePath(), capturedFace);

        }
    }


    /**
     * zapis zdjęcia
     * <p>
     * Na początku kontrola, czy istnieje folder. Jeśli nie - tworzy go
     * Stworzenie kopii zdjęcia w zmiennej greyMat klasy Mat. Przekształcenie na greyscale
     * Stworzenie wektora o 4 współrzędnych (współrzędne tworzonego kwadratu podczas detekcji twarzy), zmienna detectedFaces.
     * <p>
     * zdjęcie zapisuje się w formacie: person.personId.photonumber.jpg
     * <p>
     * użycie metody detectMultiScale() na obiekcie faceDetector:
     * detectMultiScale(greyMat, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
     * greyMat - zdjęcie z którygo chcemy wykryć twarz
     * detectedFaces - obiekt, do którego zapisujemy współrzędne, w których została wykryta twarz na zdjęciu
     * 1.1 - współczynnik, który określa o ile wielkośc obrazu zostanie zmniejszona
     * 1 - Parametr określający, ilu sąsiadów każdy kandydujący prostokąt powinien zachować.
     * 0 - używane w starszych wersjach cascadeClassifier
     * 150, 150 - minimalny rozmiar w pikselach fragmentu zdjęcia, na którym znajduje się twarz
     * 500, 500 - maksymalny rozmiar na zdjęciu, na którym może znaleźć się twarz
     * <p>
     * wykonuje się pętla przechodząca po wszystkich wykrytych tawrzach (zwykle, i poprawnie po jednej),
     * w tej pętli wszystkie wykryte twarze nadpisywane są do wcześniej stworzonego pliku .jpg w skali szarości i rozmiarze określonym
     * w zmiennej IMG_SIZE
     *
     * @param faceDetector - obiekt CascadeClassifier z pliku frontalface.xml
     */
    public static void detectFaceFromPhotos(Context context, opencv_objdetect.CascadeClassifier faceDetector, String personDirName) throws Exception {
        //File folder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File folder = new File("/mnt/sdcard/", TRAIN_FOLDER + "/" + personDirName);
        Log.d("Piopr", folder.toString());
        if (folder.exists() && !folder.isDirectory())
            folder.delete();
        if (!folder.exists())
            folder.mkdirs();

        File[] allPhotos = TrainHelper.listPhotos();
        if (allPhotos.length == 0 || allPhotos == null) {
            Toast.makeText(context, "Brak zdjec do przetworzenia", Toast.LENGTH_SHORT).show();
            return;
        }


        for (File f : allPhotos) {
            Mat photoMat = imread(f.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int height = photoMat.rows();
            int width = photoMat.cols();
            int maxFaceSize = height > width ? width : height;
            int minFaceSize = maxFaceSize / 6;
            int newWidth = 640;
            int newHeight = (int) (640 * (4.0f / 3.0f));

            if (width != 160) {
                //resize(photoMat, photoMat, new Size(newWidth, newHeight));
                opencv_core.RectVector detectedFaces = new opencv_core.RectVector();
                //faceDetector.detectMultiScale(photoMat, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
                faceDetector.detectMultiScale(photoMat, detectedFaces, 1.1, 1, 0, new Size(minFaceSize, minFaceSize), new Size(maxFaceSize, maxFaceSize));
                Log.d("Piopr", "wykrywanie twarzy na zdjeciu");
                Log.d("Piopr", "detectet size: " + detectedFaces.size());
                if(detectedFaces.size()==0){
                    if(f.delete()){
                        Log.d("Piopr", "Usunieto");
                    } else {
                        Log.d("Piopr", "Nie usunieto");
                    }
                }
                int maxRect = 0;
                for (int i = 0; i < detectedFaces.size(); i++) {
                    opencv_core.Rect rectFace = detectedFaces.get(i);
                    if(rectFace.size().area()>detectedFaces.get(maxRect).size().area()){
                        maxRect = i;
                    }
                }
                if(detectedFaces.size()>0){
                    opencv_core.Rect rectFace = detectedFaces.get(maxRect);

                    //czy zamiast photoMat oryginalny w kolorze?
                    rectangle(photoMat, rectFace, new opencv_core.Scalar(0, 0, 255, 0));


                    Mat capturedFace = new Mat(photoMat, rectFace);

                    resize(capturedFace, capturedFace, new Size(IMG_SIZE, IMG_SIZE));

                    imwrite(f.getAbsolutePath(), capturedFace);
                    //imwrite(f.getAbsolutePath(), photoMat);

                }

            }


        }
        Toast.makeText(context, "Wykryto twarze na zdjeciach.", Toast.LENGTH_SHORT).show();


    }


    public static void recognizeFromPhoto(Context context, String personDirName) throws Exception {
        Toast.makeText(context, "Sprawdzanie...", Toast.LENGTH_SHORT).show();

        IS_TRAINED = false;

        Log.d("Piopr", "Funkcja recognizeFromPhoto");
        File currentFolder = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER + "/default");
        opencv_objdetect.CascadeClassifier faceDetector;
        String[] usersNamesArray = TrainHelper.getUserNames();
        Mat detectedFace = new Mat();


        if (!currentFolder.isDirectory() || !currentFolder.exists()) {
            Toast.makeText(context, "Folder default nie istnieje", Toast.LENGTH_SHORT).show();
            currentFolder.mkdir();
            if (isTrained(context)) {
                IS_TRAINED = true;
            }
            return;
        }
        File[] photosList = currentFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File file, String s) {
                return s.endsWith(".jpg") || s.endsWith(".gif") || s.endsWith(".png");
            }
        });
        if (photosList.length == 0 || photosList == null) {
            Toast.makeText(context, "Brak zdjęcia do rozpoznania", Toast.LENGTH_SHORT).show();
            if (isTrained(context)) {
                IS_TRAINED = true;
            }
            return;
        }

        //petla wykonuje sie do pierwszego wykrycia twarzy na którymkolwiek z wylistowanych zdjęć
        loopBreaker:
        for (int i = 0; i < photosList.length; i++) {


            Mat photo = imread(photosList[i].getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            opencv_core.RectVector faces = new opencv_core.RectVector();

            int height = photo.rows();
            int width = photo.cols();
            int maxFaceSize = height > width ? width : height;
            int minFaceSize = maxFaceSize / 6;
            if (minFaceSize < 160) {
                minFaceSize = 160;
            }
            if (height == 160) {
                detectedFace = photo;
                Log.d("Piopr", "zdjecie juz gotowe");
                break loopBreaker;
            } else {

                faceDetector = loadClassifierCascade(context, R.raw.frontalface);
                faceDetector.detectMultiScale(photo, faces, 1.25f, 3, 1,
                        new Size(minFaceSize, minFaceSize),
                        new Size(maxFaceSize, maxFaceSize));


                if (faces.size() == 1) {
                    opencv_core.Rect face = faces.get(0);
                    detectedFace = new Mat(photo, face);
                    resize(detectedFace, detectedFace, new Size(IMG_SIZE, IMG_SIZE));
                    imwrite(currentFolder + "/zzzdefault.jpg", detectedFace);
                    if (detectedFace.total() == 0) {
                        Toast.makeText(context, "Nie wykryto twarzy na zdjęciu", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    break loopBreaker;
                }
            }

        }
        imwrite(currentFolder + "/zzzdefault.jpg", detectedFace);


        if (!isTrained(context) && detectedFace.total() != 0) {
            Toast.makeText(context, "Algorytm niewytrenowany", Toast.LENGTH_SHORT).show();
            return;
        }
        File trainfolder = new File("/mnt/sdcard/" + TRAIN_FOLDER);
        File eigenFile = new File(trainfolder, EIGEN_FACES_CLASSIFIER);
        File fisherFile = new File(trainfolder, FISHER_FACES_CLASSIFIER);
        File lbphFile = new File(trainfolder, LBPH_CLASSIFIER);
        String eigenText = "";
        String fisherText = "";
        String lbphText = "";
        double accLvl = 4000.00D;


        if (eigenFile.exists()) {
            FaceRecognizer eigenFaces = opencv_face.EigenFaceRecognizer.create();
            eigenFaces.read(eigenFile.getAbsolutePath());
            IntPointer label = new IntPointer(1);
            DoublePointer reliability = new DoublePointer(1);
            eigenFaces.predict(detectedFace, label, reliability);
            int prediction = label.get(0);
            double acceptanceLevel = reliability.get(0);
            if (prediction == -1 || acceptanceLevel >= accLvl) {
                eigenText = "Nierozpoznano.\n";

            } else {
                eigenText = "Witaj " + usersNamesArray[prediction] + "! " +
                        acceptanceLevel + "id: " +
                        prediction + "\n";
            }
        }


        if (fisherFile.exists()) {
            FaceRecognizer fisherFaces = opencv_face.FisherFaceRecognizer.create();
            fisherFaces.read(fisherFile.getAbsolutePath());

            IntPointer label = new IntPointer(1);
            DoublePointer reliability = new DoublePointer(1);
            fisherFaces.predict(detectedFace, label, reliability);
            int prediction = label.get(0);
            double acceptanceLevel = reliability.get(0);
            if (prediction == -1 || acceptanceLevel >= accLvl) {
                fisherText = "Nierozpoznano.\n";

            } else {
                fisherText = "Witaj " + usersNamesArray[prediction] + "! " +
                        cvRound(acceptanceLevel) + "id: " +
                        prediction + "\n";
            }
        }

        if (lbphFile.exists()) {
            FaceRecognizer lbph = opencv_face.LBPHFaceRecognizer.create();
            lbph.read(lbphFile.getAbsolutePath());
            IntPointer label = new IntPointer(1);
            DoublePointer reliability = new DoublePointer(1);
            lbph.predict(detectedFace, label, reliability);
            int prediction = label.get(0);
            double acceptanceLevel = reliability.get(0);
            if (prediction == -1 || acceptanceLevel >= accLvl) {
                lbphText = "Nierozpoznano.\n";

            } else {
                lbphText = "Witaj " + usersNamesArray[prediction] + "! " +
                        cvRound(acceptanceLevel) + "id: " +
                        prediction + "\n";
                if (CURRENT_IDUSER == prediction) {
                    VERIFIED = true;
                } else {
                    VERIFIED = false;
                }
            }
        }


        Toast.makeText(context, eigenText + fisherText + lbphText, Toast.LENGTH_SHORT).show();
        Log.d("Piopr", eigenText + fisherText + lbphText);
        if (isTrained(context)) {
            IS_TRAINED = true;
        }


    }

    public static void predictTest(Context context){
        String outputText ="";
        File mainFolder = new File("/mnt/sdcard/"+TRAIN_FOLDER);
        FileFilter directoryFilter = new FileFilter() {
            @Override
            public boolean accept(File file) {
                return file.isDirectory();
            }
        };



        FilenameFilter photosFilter = new FilenameFilter() {
            @Override
            public boolean accept(File file, String s) {
                return s.endsWith(".jpg") || s.endsWith(".gif") || s.endsWith(".png");
            }
        };
        File[] folderList = mainFolder.listFiles(directoryFilter);
        if(folderList.length==0) {
            Toast.makeText(context, "Brak stworzonych uzytkownikow.", Toast.LENGTH_SHORT).show();
            return;
        }
        File eigenFile = new File(mainFolder, EIGEN_FACES_CLASSIFIER);
        File fisherFile = new File(mainFolder, FISHER_FACES_CLASSIFIER);
        File lbphFile = new File(mainFolder, LBPH_CLASSIFIER);
        FaceRecognizer eigenFaces = opencv_face.EigenFaceRecognizer.create();
        if(eigenFile.exists()){
            eigenFaces.read(eigenFile.getAbsolutePath());
        }
        FaceRecognizer fisherFaces = opencv_face.FisherFaceRecognizer.create();
        if(fisherFile.exists()){
            fisherFaces.read(fisherFile.getAbsolutePath());
        }
        FaceRecognizer lbph = opencv_face.LBPHFaceRecognizer.create();
        if(lbphFile.exists()){
            lbph.read(lbphFile.getAbsolutePath());
        }

        Mat detectedFace = new Mat();

        List<String> eigenOutput = new ArrayList<>();
        List<String> fisherOutput = new ArrayList<>();
        List<String> lbphOutput = new ArrayList<>();
        for(File currentFolder : folderList){
            Log.d("Piopr", "Wykonanie dla: " + currentFolder.getName());
            int currentId =  Integer.parseInt(currentFolder.getName().replaceAll("[^0-9]", ""));
            File defaultFolder = new File(currentFolder, "default");
            if(!defaultFolder.exists()){
                Toast.makeText(context, "Folder default nie istnieje", Toast.LENGTH_SHORT).show();
                defaultFolder.mkdir();
            }
            File[] photosList = defaultFolder.listFiles(photosFilter);
            if (photosList.length != 0) {
                Log.d("Piopr", "Sa zdjecia");
                loopBreaker:
                for (File f : photosList) {
                    Mat photo = imread(f.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
                    opencv_core.RectVector faces = new opencv_core.RectVector();

                    int height = photo.rows();
                    int width = photo.cols();
                    int maxFaceSize = height > width ? width : height;
                    int minFaceSize = maxFaceSize / 6;
                    if (minFaceSize < 160) {
                        minFaceSize = 160;
                    }
                    if (height == 160) {
                        detectedFace = photo;
                        break loopBreaker;
                    } else {
                        opencv_objdetect.CascadeClassifier faceDetector = loadClassifierCascade(context, R.raw.frontalface);
                        faceDetector.detectMultiScale(photo, faces, 1.25f, 3, 1,
                                new Size(minFaceSize, minFaceSize),
                                new Size(maxFaceSize, maxFaceSize));
                        if (faces.size() == 1) {
                            opencv_core.Rect face = faces.get(0);
                            detectedFace = new Mat(photo, face);
                            resize(detectedFace, detectedFace, new Size(IMG_SIZE, IMG_SIZE));
                            imwrite(defaultFolder + "/zzzdefault.jpg", detectedFace);
                            Log.d("Piopr", "Wykryto w " + currentFolder.getName());
                            break loopBreaker;

                        } else {
                            detectedFace = null;
                            Log.d("Piopr", "Nie wykryto w " + currentFolder.getName());
                        }
                    }
                    if (detectedFace == null || detectedFace.total() == 0) {
                        Log.d("Piopr", "Brak twarzy");
                    }
                }
                //TODO: dokonczyc wykrywania i predykcje oraz odpowiedni id usera;
            } else {
                Log.d("Piopr", "Nie ma zdjec");
                detectedFace = null;
            }
            if(detectedFace!=null){
                if(!eigenFaces.empty()){
                    IntPointer label = new IntPointer(1);
                    DoublePointer reliability = new DoublePointer(1);
                    eigenFaces.predict(detectedFace, label, reliability);
                    int prediction = label.get(0);
                    double acceptanceLevel = reliability.get(0);
                    String output;
                    if(prediction==currentId){
                        output = currentFolder.getName()+", "+
                                "true, "+
                                acceptanceLevel;
                    } else {
                        output = currentFolder.getName()+", "+
                                "false, "+
                                acceptanceLevel;
                    }
                    eigenOutput.add(output);
                }

                if(!fisherFaces.empty()){
                    IntPointer label = new IntPointer(1);
                    DoublePointer reliability = new DoublePointer(1);
                    fisherFaces.predict(detectedFace, label, reliability);
                    int prediction = label.get(0);
                    double acceptanceLevel = reliability.get(0);
                    String output;
                    if(prediction==currentId){
                        output = currentFolder.getName()+", "+
                                "true, "+
                                acceptanceLevel;
                    } else {
                        output = currentFolder.getName()+", "+
                                "false, "+
                                acceptanceLevel;
                    }
                    fisherOutput.add(output);
                }
                if(!lbph.empty()){
                    IntPointer label = new IntPointer(1);
                    DoublePointer reliability = new DoublePointer(1);
                    lbph.predict(detectedFace, label, reliability);
                    int prediction = label.get(0);
                    double acceptanceLevel = reliability.get(0);
                    String output;
                    if(prediction==currentId){
                        output = currentFolder.getName()+", "+
                                "true, "+
                                acceptanceLevel;
                    } else {
                        output = currentFolder.getName()+", "+
                                "false, "+
                                acceptanceLevel;
                    }
                    lbphOutput.add(output);
                }

            }
            else {
                String noFaceOutput = currentFolder.getName()+", "+
                        "false, "+
                        0;
                eigenOutput.add(noFaceOutput);
                fisherOutput.add(noFaceOutput);
                lbphOutput.add(noFaceOutput);
            }



        }

        for(String out : eigenOutput){
            Log.d("Piopr", "eigen: " + out);
        }
        for(String out : fisherOutput){
            Log.d("Piopr", "fisher: "+out);
        }
        for(String out : lbphOutput){
            Log.d("Piopr", "lbph: "+out);
        }


    }




    /**
     * załadowanie kaskady do detekcji twarzy
     */
    public static opencv_objdetect.CascadeClassifier loadClassifierCascade(Context context, int resId) {
        FileOutputStream fos = null;
        InputStream inputStream;

        inputStream = context.getResources().openRawResource(resId);
        File xmlDir = context.getDir("xml", Context.MODE_PRIVATE);
        File cascadeFile = new File(xmlDir, "temp.xml");
        try {
            fos = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            Log.d(TAG, "Can\'t load the cascade file");
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        opencv_objdetect.CascadeClassifier detector = new opencv_objdetect.CascadeClassifier(cascadeFile.getAbsolutePath());
        if (detector.isNull()) {
            Log.e(TAG, "Failed to load cascade classifier");
            detector = null;
        } else {
            Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());
        }
        // delete the temporary directory
        cascadeFile.delete();

        return detector;
    }

    /***
     * Pobiera id z nazw folderów.
     * @return lista z id
     */
    public static Integer[] getUserIds() {
        File trainFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] users = trainFolder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });
        Integer[] usersIds = new Integer[users.length];
        for (int i = 0; i < users.length; i++) {
            usersIds[i] = Integer.parseInt(users[i].replaceAll("[^0-9]", ""));
        }
        return usersIds;
    }

    /***
     * Listuję liste uzytkownikow.
     * @return Tablica stringow z nazwami uzyktownikow.
     */
    public static String[] getUserNames() {
        File trainFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] users = trainFolder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });
        for (int i = 0; i < users.length; i++) {
            users[i] = users[i].replaceAll("[0-9]","");
        }
        Log.d("Piopr", "Dlogosc users: " + users.length);
        String[] usersList = new String[users.length + 1];
        Log.d("Piopr", "Dlogosc usersList: " + usersList.length);
        usersList[0] = "";
        for (int i = 0; i < users.length; i++) {
            usersList[i + 1] = users[i];
        }
        Log.d("Piopr", "Dlogosc usersList2: " + usersList.length);

        return usersList;
    }

    /**
     * Listuje wszystkie zdjecia znajdujace sie w folderze aktualnego uzytkownika.
     *
     * @return - tablica String[] zawierająca wszyskie pliki zdjęciowe w folderze.
     */
    public static File[] listPhotos() {
        File currentFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER);

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        File[] allPhotosArray = currentFolder.listFiles(imageFilter);
        Log.d("Piopr", "Dlugosc allPhotos: " + allPhotosArray.length);
//        if(allPhotosArray.length != 0 || allPhotosArray != null) {
//            Log.d("Piopr", "Sciezka losowego zdjecia: " + allPhotosArray[0].getAbsolutePath());
//        }

        return allPhotosArray;
    }

    /**
     * Zmienia nazwy zdjec odpowiednio do patternu person.id.numerZdjecia
     */
    public static void renamePhotos() {

        int photoNr = 0;

        File currentFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER);

        FilenameFilter imageFilter = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        File[] allPhotosArray = currentFolder.listFiles(imageFilter);
        int i = 0;
        for (File f : allPhotosArray) {
            i++;
            File name = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER, "tmp" + i + ".jpg");
            f.renameTo(name);
        }

        allPhotosArray = currentFolder.listFiles(imageFilter);

        for (File f : allPhotosArray) {
            photoNr++;
            File newName = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER, String.format(FILE_NAME_PATTERN, TrainHelper.CURRENT_IDUSER, photoNr));

            if (f.renameTo(newName)) {
                Log.d("Piopr", "Zmieniono: " + f.getAbsolutePath());

            } else {

                Log.d("Piopr", "Nie zmieniono: " + f.getAbsolutePath());
            }

            Log.d("Piopr", "Nowa nazwa pliku: " + newName.getAbsolutePath());

        }


        Log.d("Piopr", "Zmieniononazwy plikow");


    }

    /***
     * Sprawdza, czy istnieje więcej, niż jeden użytkownik do treningu (jego zestaw zdjęć).
     * Sprawdzenie potrzebne do algorytmu FisherFaces, który potrzebuje przynajmniej dwoch zestawów.
     * @param idBuffer - Mat labels potrzebnym do treningu, w którym znajdują się id użytkownikow
     * @return true: jeśli istnieje wiecej, niż jeden,  false: jeśli tylko jeden zestaw zdjęć
     *
     */


    public static boolean checkCouplePersonsExists(IntIndexer idBuffer) {
        int valueToCompare = idBuffer.get(0, 0);
        for (int i = 0; i < idBuffer.rows(); i++) {
            if (valueToCompare != idBuffer.get(i, 0))
                return true;
        }

        return false;
    }


    public static boolean checkFisherExists() {
        File file = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + FISHER_FACES_CLASSIFIER);
        return file.exists();
    }


}