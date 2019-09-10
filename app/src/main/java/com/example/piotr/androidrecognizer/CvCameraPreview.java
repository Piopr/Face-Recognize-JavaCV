/*
 * Copyright 2007-2011 the original author or authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.piotr.androidrecognizer;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.FFmpegFrameFilter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameFilter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

import static org.bytedeco.javacpp.avutil.AV_PIX_FMT_NV21;

public class CvCameraPreview extends SurfaceView implements SurfaceHolder.Callback, PreviewCallback {

    /**
     * LOG_TAG - nazwa tagu do logowania
     */
    private final String LOG_TAG = "CvCameraPreview";

    private static final int STOPPED = 0;
    private static final int STARTED = 1;

    /**
     * id wybranej kamery
     */
    public static final int CAMERA_BACK = 99;
    public static final int CAMERA_FRONT = 98;

    public static final int SCALE_FIT = 1;
    public static final int SCALE_FULL = 2;

    private static final int MAGIC_TEXTURE_ID = 10;

    /**
     * ASPECT_RATIO_W and ASPECT_RATIO_H define the aspect ratio
     * of the Surface. They are used when {@link #onMeasure(int, int)}
     * is called.
     * Wspolczynniki proporcji, 4:3
     */
    private final float ASPECT_RATIO_W = 4.0f;
    private final float ASPECT_RATIO_H = 3.0f;

    /**
     * The maximum dimension (in pixels) of the preview frames that are produced
     * by the Camera object. Note that this should not be intended as
     * the final, exact, dimension because the device could not support
     * it and a lower value is required (but the aspect ratio should remain the same).<br />
     * See {@link CvCameraPreview#getBestSize(List, int)} for more information.
     */
    private final int PREVIEW_MAX_WIDTH = 640;

    /**
     * rozdzielczosc zdjecia zrobionego przez kamere
     * The maximum dimension (in pixels) of the images produced when a
     * {@link PictureCallback#onPictureTaken(byte[], Camera)} event is
     * fired. Again, this is a maximum value and could not be the
     * real one implemented by the device.
     */
    private final int PICTURE_MAX_WIDTH = 1280;

    /**
     * bufor podglądu na którym dokonujemy obliczeń lub transformacji
     * In this example we look at camera preview buffer functionality too.<br />
     * This is the array that will be filled everytime a single preview frame is
     * ready to be processed (for example when we want to show to the user
     * a transformed preview instead of the original one, or when we want to
     * make some image analysis in real-time without taking full-sized pictures).
     */
    private byte[] previewBuffer;

    /**
     * //uchwyt pozwalający na modyfikację aktualnie wyświetlanego podglądu
     * The "holder" is the underlying surface.
     */
    private SurfaceHolder surfaceHolder;
    /**
     * narzędzie służące do filtrowania ramek
     */
    private FFmpegFrameFilter filter;
    /**
     * id łańcucha ???
     */
    private int chainIdx = 0;
    /**
     * mutexy (zmienne do wstrzymywania wątków)
     */
    private boolean stopThread = false;
    private boolean cameraFrameReady = false;
    protected boolean enabled = true;
    private boolean surfaceExist;
    /**
     * zmienna wątku
     */
    private Thread thread;
    /**
     * interfejs obsługi zmiany podglądu i dostarczania ramek
     */
    private CvCameraViewListener listener;
    /**
     * klasa konwersji obiektów klasy Frame do Bitmapy
     */
    private AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
    /**
     * konwerjsa bitmapy do obiektu typu Mat (obliczenia w JavaCV wykonywane właśnie na tej klasie).
     */
    private OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
    /**
     * pamięć podręczna bitmapy
     */
    private Bitmap cacheBitmap;
    /**
     * tablica ramek
     */
    protected Frame[] cameraFrame;
    /**
     * stan.     *
     */
    private int state = STOPPED;
    /**
     * obiekt do współbieżności
     */
    private final Object syncObject = new Object();
    /**
     * zmienna id kamery (wybór pomiędzy dostępnymi kamerami)
     */
    private int cameraId = -1;
    /**
     * pobieranie id tylnej kamery
     */
    private int cameraType = Camera.CameraInfo.CAMERA_FACING_BACK;
    /**
     * klasa służy do sterowania kamerą (ustawienia podglądu, zatrzymywanie, wyświetlanie)
     */
    private Camera cameraDevice;
    /**
     * przechowuje ramki ze strumienia obrazu jako tekst ES OpenGL (pochodzi z kamery lub z wideo)
     * updateTexImage () - aktualizuje na najnowszą klatkę ze źródła obrazu
     */
    private SurfaceTexture surfaceTexture;
    /**
     * rozmiary klatek
     */
    private int frameWidth, frameHeight;
    /**
     * skalowanie do aktualnego widoku (dopasowanie)
     */
    private int scaleType = SCALE_FIT;

    /**
     * dziedziczyz SurfaceView
     * @param context - aktualny widok
     * @param attrs - atrybuty (pobierane z opencv.xml)
     */
    public CvCameraPreview(Context context, AttributeSet attrs) {
        super(context, attrs);

        /**
         * pobieranie wartości z pliku /values/attrs.xml
         */
        TypedArray array = getContext().obtainStyledAttributes(attrs, R.styleable.CvCameraPreview);
        int camType = array.getInt(R.styleable.CvCameraPreview_camera_type, CAMERA_BACK);
        int scaleType = array.getInt(R.styleable.CvCameraPreview_scale_type, SCALE_FIT);
        array.recycle();

        initializer(camType == CAMERA_BACK ? Camera.CameraInfo.CAMERA_FACING_BACK : Camera.CameraInfo.CAMERA_FACING_FRONT, scaleType);

    }

    public CvCameraPreview(Context context, int camType, int scaleType) {
        super(context);

        initializer(camType == CAMERA_BACK ? Camera.CameraInfo.CAMERA_FACING_BACK : Camera.CameraInfo.CAMERA_FACING_FRONT, scaleType);
    }

    private void initializer(int camType, int scaleType) {

        this.surfaceHolder = this.getHolder();
        this.surfaceHolder.addCallback(this);

        this.scaleType = scaleType;
        this.cameraType = camType;

        if (Build.VERSION.SDK_INT < 11) {
            this.surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        }
    }

    /**
     * listener zmiany podglądu i dostarczania ramek
     * @param listener - zmiana listenera
     */
    public void setCvCameraViewListener(CvCameraViewListener listener) {
        this.listener = listener;
    }

    /**
     *
     * @return - id uzywanej kamery
     */
    public int getCameraId() {
        return cameraId;
    }

    /**
     * Called when the surface is created for the first time. It sets all the
     * required {@link #cameraDevice}'s parameters and starts the preview stream.
     *
     * @param holder
     */
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    /**
     * [IMPORTANT!] A SurfaceChanged event means that the parent graphic has changed its layout
     * (for example when the orientation changes). It's necessary to update the {@link CvCameraPreview}
     * orientation, so the preview is stopped, then updated, then re-activated.
     *
     * @param holder The SurfaceHolder whose surface has changed
     * @param format The new PixelFormat of the surface
     * @param w      The new width of the surface
     * @param h      The new height of the surface
     */
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        if (this.surfaceHolder.getSurface() == null) {

            return;
        }

        synchronized (syncObject) {
            if (!surfaceExist) {
                surfaceExist = true;
                checkCurrentState();
            } else {
                surfaceExist = false;
                checkCurrentState();

                surfaceExist = true;
                checkCurrentState();
            }
        }
    }

    /**
     * Called when syncObject lock is held
     */
    private void checkCurrentState() {

        int targetState;

        if (enabled && surfaceExist && getVisibility() == VISIBLE) {
            targetState = STARTED;
        } else {
            targetState = STOPPED;
        }

        if (targetState != state) {
            /* The state change detected. Need to exit the current state and enter target state */
            processExitState(state);
            state = targetState;
            processEnterState(state);
        }
    }

    private void processExitState(int state) {

        switch (state) {
            case STARTED:
                onExitStartedState();
                break;
            case STOPPED:
                onExitStoppedState();
                break;
        }
        ;
    }

    private void processEnterState(int state) {

        switch (state) {
            case STARTED:
                /**
                 * sprawdzenie, czy kamera moze byc uzywana. Jesli nie, przerywa aplikacje
                 */
                onEnterStartedState();
                if (listener != null) {
                    listener.onCameraViewStarted(frameWidth, frameHeight);
                }
                break;
            case STOPPED:
                onEnterStoppedState();
                if (listener != null) {
                    listener.onCameraViewStopped();
                }
                break;
        }
    }

    private void onEnterStoppedState() {
        /* nothing to do */
    }

    private void onExitStoppedState() {
        /* nothing to do */
    }

    // NOTE: The order of bitmap constructor and camera connection is important for android 4.1.x
    // Bitmap must be constructed before surface
    private void onEnterStartedState() {

        /* Connect camera */
        if (!connectCamera()) {
            AlertDialog ad = new AlertDialog.Builder(getContext()).create();
            ad.setCancelable(false); // This blocks the 'BACK' button
            ad.setMessage("Kamera nie jest wspierana. Zamykanie aplikacji.");
            ad.setButton(DialogInterface.BUTTON_NEUTRAL, "OK", new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                    ((Activity) getContext()).finish();
                }
            });
            ad.show();

        }
    }

    private void onExitStartedState() {
        disconnectCamera();
        if (cacheBitmap != null) {
            cacheBitmap.recycle();
        }
        if (filter != null) {
            try {
                /**
                 * zamyka aktualny plik wideo lub urządzenie
                 */
                filter.release();
            } catch (FrameFilter.Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * obsluga zamkniecia powierzchni surfaceHolder
     * @param holder
     */
    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

        synchronized (syncObject) {
            surfaceExist = false;
            checkCurrentState();
        }
    }

    /**
     * [IMPORTANT!] Probably the most important method here. Lots of users experience bad
     * camera behaviors because they don't override this guy.
     * In fact, some Android devices are very strict about the size of the surface
     * where the preview is printed: if its ratio is different from the
     * original one, it results in errors like "startPreview failed".<br />
     * This methods takes care on this and applies the right size to the
     * {@link CvCameraPreview}.
     *
     * @param widthMeasureSpec  horizontal space requirements as imposed by the parent.
     * @param heightMeasureSpec vertical space requirements as imposed by the parent.
     */
    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int height = MeasureSpec.getSize(heightMeasureSpec);
        int width = MeasureSpec.getSize(widthMeasureSpec);

        // do some ultra high precision math...
        /**
         * obliczenie dokładnej proporcji w celu uniknięcia błęów
         */
        float expectedRatio = height > width ? ASPECT_RATIO_H / ASPECT_RATIO_W : ASPECT_RATIO_W / ASPECT_RATIO_H;
        float screenRatio = width * 1f / height;
        if (screenRatio > expectedRatio) {
            if (scaleType == SCALE_FULL) {
                height = (int) (width / expectedRatio + .5);
            } else {
                width = (int) (height * expectedRatio + .5);
            }
        } else {
            if (scaleType == SCALE_FULL) {
                width = (int) (height * expectedRatio + .5);
            } else {
                height = (int) (width / expectedRatio + .5);
            }
        }

        setMeasuredDimension(width, height);

    }

    private boolean connectCamera() {
        /* 1. We need to instantiate camera
         * 2. We need to start thread which will be getting frames
         */
        /* First step - initialize camera connection */

        if (!initializeCamera())
            return false;

        /* now we can start update thread */

        stopThread = false;
        thread = new Thread(new CameraWorker());
        thread.start();

        return true;
    }

    private void disconnectCamera() {

        try {
            stopThread = true;

            synchronized (this) {
                this.notify();
            }

            if (thread != null)
                thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            thread = null;
        }

        stopCameraPreview();

        /* Now release camera */
        releaseCamera();

        cameraFrameReady = false;
    }

    private boolean initializeCamera() {
        synchronized (this) {
            if (this.cameraDevice != null) {
                // do the job only if the camera is not already set

                return true;
            }

            // warning here! starting from API 9, we can retrieve one from the multiple
            // hardware cameras (ex. front/back)
            if (Build.VERSION.SDK_INT >= 9) {

                if (this.cameraId < 0) {
                    // at this point, it's the first time we request for a camera
                    Camera.CameraInfo camInfo = new Camera.CameraInfo();
                    for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
                        Camera.getCameraInfo(i, camInfo);

                        if (camInfo.facing == cameraType) {
                            // in this example we'll request specifically the back camera
                            try {
                                this.cameraDevice = Camera.open(i);
                                this.cameraId = i; // assign to cameraId this camera's ID (O RLY?)
                                break;
                            } catch (RuntimeException e) {
                                // something bad happened! this camera could be locked by other apps

                            }
                        }
                    }
                } else {
                    // at this point, a previous camera was set, we try to re-instantiate it
                    try {
                        this.cameraDevice = Camera.open(this.cameraId);
                    } catch (RuntimeException e) {

                    }
                }
            }

            // we could reach this point in two cases:
            // - the API is lower than 9
            // - previous code block failed
            // hence, we try the classic method, that doesn't ask for a particular camera
            if (this.cameraDevice == null) {
                try {
                    this.cameraDevice = Camera.open();
                    this.cameraId = 0;
                } catch (RuntimeException e) {

                    return false;
                }
            }



            setupCamera();

            updateCameraDisplayOrientation();

            initFilter(frameWidth, frameHeight);

            startCameraPreview(this.surfaceHolder);
        }

        return true;
    }

    /**
     * It sets all the required parameters for the Camera object, like preview
     * and picture size, format, flash modes and so on.
     * In this particular example it initializes the {@link #previewBuffer} too.
     */
    private boolean setupCamera() {
        if (cameraDevice == null) {

            return false;
        }
        try {
            Camera.Parameters parameters = cameraDevice.getParameters();
            List<Size> sizes = parameters.getSupportedPreviewSizes();
            if (sizes != null) {
                Size bestPreviewSize = getBestSize(sizes, PREVIEW_MAX_WIDTH);

                frameWidth = bestPreviewSize.width;
                frameHeight = bestPreviewSize.height;

                parameters.setPreviewSize(bestPreviewSize.width, bestPreviewSize.height);

                parameters.setPreviewFormat(ImageFormat.NV21); // NV21 is the most supported format for preview frames

                List<String> FocusModes = parameters.getSupportedFocusModes();
                if (FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                    parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                }

                cameraDevice.setParameters(parameters); // save everything

                // print saved parameters
                int prevWidth = cameraDevice.getParameters().getPreviewSize().width;
                int prevHeight = cameraDevice.getParameters().getPreviewSize().height;
                int picWidth = cameraDevice.getParameters().getPictureSize().width;
                int picHeight = cameraDevice.getParameters().getPictureSize().height;


                // here: previewBuffer initialization. It will host every frame that comes out
                // from the preview, so it must be big enough.
                // After that, it's linked to the camera with the setCameraCallback() method.
                try {
                    this.previewBuffer = new byte[prevWidth * prevHeight * ImageFormat.getBitsPerPixel(cameraDevice.getParameters().getPreviewFormat()) / 8];
                    setCameraCallback();
                } catch (IOException e) {

                }

                return true;
            } else {
                return false;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    protected void releaseCamera() {
        synchronized (this) {
            if (cameraDevice != null) {
                cameraDevice.release();
            }
            cameraDevice = null;
        }
    }

    /**
     * [IMPORTANT!] Sets the {@link #previewBuffer} to be the default buffer where the
     * preview frames will be copied. Also sets the callback function
     * when a frame is ready.
     *
     * @throws IOException
     */
    private void setCameraCallback() throws IOException {

        cameraDevice.addCallbackBuffer(this.previewBuffer);
        cameraDevice.setPreviewCallbackWithBuffer(this);
    }

    /**
     * [IMPORTANT!] This is a convenient function to determine what's the proper
     * preview/picture size to be assigned to the camera, by looking at
     * the list of supported sizes and the maximum value given
     *
     * @param sizes          sizes that are currently supported by the camera hardware,
     *                       retrived with {@link Camera.Parameters#getSupportedPictureSizes()} or {@link Camera.Parameters#getSupportedPreviewSizes()}
     * @param widthThreshold the maximum value we want to apply
     * @return an optimal size <= widthThreshold
     */
    private Size getBestSize(List<Size> sizes, int widthThreshold) {
        Size bestSize = null;

        for (Size currentSize : sizes) {
            boolean isDesiredRatio = ((currentSize.width / ASPECT_RATIO_W) == (currentSize.height / ASPECT_RATIO_H));
            boolean isBetterSize = (bestSize == null || currentSize.width > bestSize.width);
            boolean isInBounds = currentSize.width <= widthThreshold;

            if (isDesiredRatio && isInBounds && isBetterSize) {
                bestSize = currentSize;
            }
        }

        if (bestSize == null) {
            bestSize = sizes.get(0);

        }


        return bestSize;
    }

    /**
     * In addition to calling {@link Camera#startPreview()}, it also
     * updates the preview display that could be changed in some situations
     *
     * @param holder the current {@link SurfaceHolder}
     */
    private synchronized void startCameraPreview(SurfaceHolder holder) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
                surfaceTexture = new SurfaceTexture(MAGIC_TEXTURE_ID);
                cameraDevice.setPreviewTexture(surfaceTexture);
            } else
                cameraDevice.setPreviewDisplay(holder);
            cameraDevice.startPreview();
//            filter.start();
        } catch (Exception e) {

        }
    }

    /**
     * It "simply" calls {@link Camera#stopPreview()} and checks
     * for errors
     */
    private synchronized void stopCameraPreview() {
        try {
            cameraDevice.stopPreview();
            cameraDevice.setPreviewCallback(null);
//            filter.stop();
        } catch (Exception e) {
            // ignored: tried to stop a non-existent preview

        }
    }

    /**
     * Gets the current screen rotation in order to understand how much
     * the surface needs to be rotated
     */
    private void updateCameraDisplayOrientation() {
        if (cameraDevice == null) {

            return;
        }

        int degree = getRotationDegree();

        cameraDevice.setDisplayOrientation(degree); // save settings
    }

    private int getRotationDegree() {
        int result = 0;
        Activity parentActivity = (Activity) this.getContext();

        int rotation = parentActivity.getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        if (Build.VERSION.SDK_INT >= 9) {
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(cameraId, info);

            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                result = (info.orientation + degrees) % 360;
                result = (360 - result) % 360;  // compensate the mirror
            } else {
                // back-facing
                result = (info.orientation - degrees + 360) % 360;
            }
        } else {
            result = Math.abs(degrees - 90);
        }
        return result;
    }

    private void initFilter(int width, int height) {
        int degree = getRotationDegree();
        Camera.CameraInfo info = new Camera.CameraInfo();
        Camera.getCameraInfo(cameraId, info);
        boolean isFrontFaceCamera = info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT;

        String transposeCode;
        String formatCode = "format=pix_fmts=rgba";
        /*
         0 = 90CounterCLockwise and Vertical Flip (default)
         1 = 90Clockwise
         2 = 90CounterClockwise
         3 = 90Clockwise and Vertical Flip
         */
        switch (degree) {
            case 0:
                transposeCode = isFrontFaceCamera ? "transpose=3,transpose=2" : "transpose=1,transpose=2";
                break;
            case 90:
                transposeCode = isFrontFaceCamera ? "transpose=3" : "transpose=1";
                break;
            case 180:
                transposeCode = isFrontFaceCamera ? "transpose=0,transpose=2" : "transpose=2,transpose=2";
                break;
            case 270:
                transposeCode = isFrontFaceCamera ? "transpose=0" : "transpose=2";
                break;
            default:
                transposeCode = isFrontFaceCamera ? "transpose=3,transpose=2" : "transpose=1,transpose=2";
        }

        if (cameraFrame == null) {
            cameraFrame = new Frame[2];
            cameraFrame[0] = new Frame(width, height, Frame.DEPTH_UBYTE, 2);
            cameraFrame[1] = new Frame(width, height, Frame.DEPTH_UBYTE, 2);
        }

        filter = new FFmpegFrameFilter(transposeCode + "," + formatCode, width, height);
        filter.setPixelFormat(AV_PIX_FMT_NV21);

        Log.i(LOG_TAG, "inicjalizacja filtra");
    }

    @Override
    public void onPreviewFrame(byte[] raw, Camera cam) {
        processFrame(previewBuffer, cam);
        cam.addCallbackBuffer(previewBuffer);
    }

    /**
     * @param raw the preview buffer
     * @param cam the camera that filled the buffer
     * @see <a href="http://en.wikipedia.org/wiki/YUV#Y.27UV420sp_.28NV21.29_to_ARGB8888_conversion">YUV Conversion - Wikipedia</a>
     */
    private void processFrame(byte[] raw, Camera cam) {
        if (cameraFrame != null) {
            synchronized (this) {
                ((ByteBuffer) cameraFrame[chainIdx].image[0].position(0)).put(raw);
                cameraFrameReady = true;
                this.notify();
            }
        }
    }

    private class CameraWorker implements Runnable {
        public void run() {
            do {
                boolean hasFrame = false;
                synchronized (CvCameraPreview.this) {
                    try {
                        while (!cameraFrameReady && !stopThread) {
                            CvCameraPreview.this.wait();
                        }
                    } catch (InterruptedException e) {

                    }
                    if (cameraFrameReady) {
                        chainIdx = 1 - chainIdx;
                        cameraFrameReady = false;
                        hasFrame = true;
                    }
                }

                if (!stopThread && hasFrame) {
                    if (cameraFrame[1 - chainIdx] != null) {
                        try {
                            Frame frame;
                            filter.start();
                            filter.push(cameraFrame[1 - chainIdx]);
                            while ((frame = filter.pull()) != null) {
                                deliverAndDrawFrame(frame);
                            }
                            filter.stop();
                        } catch (FrameFilter.Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            } while (!stopThread);

        }
    }

    /**
     * @param frame - ramkma do dostarczenia
     */
    protected void deliverAndDrawFrame(Frame frame) {
        Mat processedMat = null;

        if (listener != null) {
            Mat mat = converterToMat.convert(frame);
            processedMat = listener.onCameraFrame(mat);
            frame = converterToMat.convert(processedMat);
            if (mat != null) {
                mat.release();
            }
        }
        cacheBitmap = converterToBitmap.convert(frame);
        if (cacheBitmap != null) {
            int width, height;
            Canvas canvas = getHolder().lockCanvas();
            if (canvas != null) {
                width = canvas.getWidth();
                height = cacheBitmap.getHeight() * canvas.getWidth() / cacheBitmap.getWidth();
                canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
                canvas.drawBitmap(cacheBitmap,
                        new Rect(0, 0, cacheBitmap.getWidth(), cacheBitmap.getHeight()),
                        new Rect(0,
                                (canvas.getHeight() - height) / 2,
                                width,
                                (canvas.getHeight() - height) / 2 + height), null);
                getHolder().unlockCanvasAndPost(canvas);
            }
        }

        if (processedMat != null) {
            processedMat.release();
        }
    }

    public interface CvCameraViewListener {
        /**
         * @param width  -  the width of the frames that will be delivered
         * @param height - the height of the frames that will be delivered
         */
        public void onCameraViewStarted(int width, int height);

        /**
         * wywoływana, gdy potrzebne dostarczenie ramki. Zwraca zmodyfikowaną ramkę
         */
        public Mat onCameraFrame(Mat mat);

        /**
         * metoda jest wywoływana, gdy podgląd kamery został przerwany
         * This method is invoked when camera preview has been stopped for some reason.
         * No frames will be delivered via onCameraFrame() callback after this method is called.
         */
        public void onCameraViewStopped();
    }

}
