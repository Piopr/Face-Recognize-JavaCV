/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.example.piotr.androidrecognizer;

import android.content.Context;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.support.v4.content.res.TypedArrayUtils;
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

import org.bytedeco.javacpp.indexer.IntBufferIndexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_objdetect;


import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

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

    public static String CURRENT_USER;
    public static int CURRENT_IDUSER;
    public static String CURRENT_FOLDER;
    public static boolean FISHER_EXISTS;

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
        IntBufferIndexer idBuffer = labels.createIndexer();


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

//       Mat eigenValues = ((opencv_face.EigenFaceRecognizer) eigenfaces).getEigenValues();
//       Mat eigenVectors = ((opencv_face.EigenFaceRecognizer) eigenfaces).getEigenVectors();
//       Mat mean = ((opencv_face.EigenFaceRecognizer) eigenfaces).getMean();

//        File tmpFile = new File("/mnt/sdcard/"+TRAIN_FOLDER+"/"+CURRENT_FOLDER+"/"+"default/"+"mean.jpg");
//        imwrite(tmpFile.getAbsolutePath(), mean.reshape(1, photos.get(0).rows()));
//        tmpFile = new File("/mnt/sdcard/"+TRAIN_FOLDER+"/"+CURRENT_FOLDER+"/"+"default/"+"eigenVectors.jpg");
//        Mat output = eigenVectors.reshape(1, photos.get(0).rows());
//        Log.d("Piopr", "Channels: "+ output.channels());
        //cvtColor(output, output, CV_IMWRITE_PAM_FORMAT_GRAYSCALE);
        //imwrite(tmpFile.getAbsolutePath(), output);


//        // eigenfaces.g
//       MatVector matVectofEigen = new MatVector(((opencv_face.EigenFaceRecognizer) eigenfaces).getProjections());
//        MatVector matVectofEigen2 = ((opencv_face.EigenFaceRecognizer) eigenfaces).getProjections();
//
//       //Mat matProjection = new Mat(size);
//       Mat matProjection = new Mat();
//       matProjection = matVectofEigen2.get(0);
//       Log.d("Piopr ", "matProjection rows: " + matProjection.rows()+ " cols: " +matProjection.cols());
//       Log.d("Piopr", "mat: " + matProjection.toString());
//       Log.d("Piopr", "mat: " + matProjection);

//        Mat meanEigen = ((opencv_face.EigenFaceRecognizer) eigenfaces).getMean();
//        Mat newMat = new Mat();
//        resize(meanEigen, newMat, new Size(160, 160));
//        File tmpFile = new File("/mnt/sdcard/"+TRAIN_FOLDER+"/"+CURRENT_FOLDER+"/"+"default/"+"mean1.jpg");
//        imwrite(tmpFile.getAbsolutePath(), newMat);
//        //cv2
//

//
//
//
//       File tmpFile = new File("/mnt/sdcard/"+TRAIN_FOLDER+"/"+CURRENT_FOLDER+"/"+"default/"+"zprojection1.jpg");
//       imwrite(tmpFile.getAbsolutePath(), matProjection);
//
//       Mat matProjection2 = matVectofEigen.get(12);
//       tmpFile = new File("/mnt/sdcard/"+TRAIN_FOLDER+"/"+CURRENT_FOLDER+"/"+"default/"+"zprojection2.jpg");
//        imwrite(tmpFile.getAbsolutePath(), matProjection2);


        if (checkCouplePersonsExists(idBuffer)) {
            fisherfaces.train(photos, labels);
            f = new File(photosFolder, FISHER_FACES_CLASSIFIER);
            f.createNewFile();
            fisherfaces.save(f.getAbsolutePath());
        } else {
            Toast.makeText(context, "Aby wytrenować algorytm Fisherfaces potrzeba przynajmniej dwóch zestawow zdjec", Toast.LENGTH_SHORT).show();

        }

        SystemClock.sleep(900);

        lbph.train(photos, labels);
        f = new File(photosFolder, LBPH_CLASSIFIER);
        f.createNewFile();
        lbph.save(f.getAbsolutePath());


        return true;
    }


    public static void makeMeanOfEigens(Context context) throws Exception {
        File photosFolder = new File("/mnt/sdcard/", TRAIN_FOLDER);
        if (!photosFolder.exists()) return;

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
            Log.d("Piopr", "Numer id: " + classe);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            photos.put(counter, photo);
            rotulosBuffer.put(counter, classe);
            counter++;
        }

        Log.d("Piopr", "Jeden");
        int NUM_EIGEN_FACES = 14;
        Size imgsize = photos.get(0).size();
        Mat meanFaces = new Mat(photos);
        opencv_core.PCA pca = new opencv_core.PCA(meanFaces, new Mat(), opencv_core.PCA.DATA_AS_ROW, 14);
        Mat averageFace = pca.mean().reshape(1, imgsize.height());
        Mat eigenVectors = pca.eigenvectors();

        Log.d("Piopr", "Dwa");


        for (int i = 0; i < NUM_EIGEN_FACES; i++) {
            Mat eigenFace = eigenVectors.row(i).reshape(1, imgsize.height());
            eigenFace.push_back(eigenFace);
        }

        Log.d("Piopr", "Trzy");
        Mat output = new Mat();
        resize(averageFace, output, new Size(160, 160));
        File tmpFile = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + CURRENT_FOLDER + "/" + "default/" + "mean1.jpg");
        imwrite(tmpFile.getAbsolutePath(), output);
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
    public static void takePhoto(Context context, int personId, int photoNumber, Mat rgbaMat, opencv_objdetect.CascadeClassifier faceDetector) throws Exception {
        //File folder = new File(context.getFilesDir(), TRAIN_FOLDER);
        File folder = new File("/mnt/sdcard/", TRAIN_FOLDER);
        Log.d("Piopr", folder.toString());
        if (folder.exists() && !folder.isDirectory())
            folder.delete();
        if (!folder.exists())
            folder.mkdirs();

        int qtyPhotos = PHOTOS_TRAIN_QTY;
        Mat greyMat = new Mat(rgbaMat.rows(), rgbaMat.cols());

        cvtColor(rgbaMat, greyMat, CV_BGR2GRAY);
        opencv_core.RectVector detectedFaces = new opencv_core.RectVector();
        faceDetector.detectMultiScale(greyMat, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
        for (int i = 0; i < detectedFaces.size(); i++) {
            Log.d("Piopr", "Wykonanie petli nr das das :  " + 1);
            Log.d("Piopr", "detectedFaces :  " + detectedFaces.get(0));


            opencv_core.Rect rectFace = detectedFaces.get(0);
            Log.d("Piopr", "rectFace :  " + rectFace.get());
            rectangle(rgbaMat, rectFace, new opencv_core.Scalar(0, 0, 255, 0));


            Mat capturedFace = new Mat(greyMat, rectFace);
            resize(capturedFace, capturedFace, new Size(IMG_SIZE, IMG_SIZE));

            if (photoNumber <= qtyPhotos) {
                File f = new File(folder, String.format(FILE_NAME_PATTERN, personId, photoNumber));
                f.createNewFile();
                imwrite(f.getAbsolutePath(), capturedFace);
            }


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
     * @param context      - aktualny widok
     * @param personId     - id osoby - dodawane do nazwy pliku
     * @param photoNumber  - numer zrobionego zdjęcia, dodawane do nazwy pliku
     * @param rgbaMat      - oryginalny obrazek, przechwycony z cameraactivity
     * @param faceDetector - obiekt CascadeClassifier z pliku frontalface.xml
     */
    public static void takePhotoNew(Context context, int personId, int photoNumber, Mat rgbaMat, opencv_objdetect.CascadeClassifier faceDetector, String personDirName) throws Exception {
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
        faceDetector.detectMultiScale(greyMat, detectedFaces, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
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
                for (int i = 0; i < detectedFaces.size(); i++) {
                    opencv_core.Rect rectFace = detectedFaces.get(0);

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


    public static Mat getPhotoToRecognize(Context context, String personDirName) {
        Log.d("Piopr", "Funkcja getPhotoToRecognize");
        File currentFolder = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + personDirName + "/default");
        opencv_objdetect.CascadeClassifier faceDetector;


        if (!currentFolder.isDirectory() || !currentFolder.exists()) {
            Toast.makeText(context, "Folder default nie istnieje", Toast.LENGTH_SHORT).show();
            currentFolder.mkdir();
            return null;
        }
        File[] photosList = currentFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File file, String s) {
                return s.endsWith(".jpg") || s.endsWith(".gif") || s.endsWith(".png");
            }
        });
        if (photosList.length == 0 || photosList == null) {
            Toast.makeText(context, "Brak zdjęcia do rozpoznania", Toast.LENGTH_SHORT).show();
            return null;
        }

        Mat photo = imread(photosList[0].getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
        opencv_core.RectVector faces = new opencv_core.RectVector();

        int height = photo.rows();
        int width = photo.cols();
        int maxFaceSize = height > width ? width : height;
        int minFaceSize = maxFaceSize / 6;


        faceDetector = loadClassifierCascade(context, R.raw.frontalface);
        faceDetector.detectMultiScale(photo, faces, 1.25f, 3, 1,
                new Size(minFaceSize, minFaceSize),
                new Size(maxFaceSize, maxFaceSize));

        if (faces.size() == 1) {
            opencv_core.Rect face = faces.get(0);
            photo = new Mat(photo, face);
            resize(photo, photo, new Size(IMG_SIZE, IMG_SIZE));
            imwrite(photosList[0].getAbsolutePath() + "1.jpg", photo);

        } else {
            Toast.makeText(context, "Nie wykryto twarzy na zdjęciu", Toast.LENGTH_SHORT).show();
        }

        if (!isTrained(context)) {
            Toast.makeText(context, "Algorytm niewytrenowany", Toast.LENGTH_SHORT).show();
        }

        FaceRecognizer eigenfaces = opencv_face.EigenFaceRecognizer.create();


        return null;
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
            usersIds[i] = Integer.parseInt(users[i].substring(0, 1));
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
            users[i] = users[i].substring(1);
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
        for (File f : allPhotosArray){
            i++;
            File name = new File("/mnt/sdcard/" + TRAIN_FOLDER + "/" + TrainHelper.CURRENT_FOLDER, "tmp"+i+".jpg");
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
    public static boolean checkCouplePersonsExists(IntBufferIndexer idBuffer) {
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