package com.example.piotr.androidrecognizer;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;

import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private boolean mPermissionReady; //zmienna do kontroli nadanych uprawnien

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        Log.d("Piopr", "dzialaj no");
        setContentView(R.layout.activity_main); //ustawienie widoku
        //obsluga klikniecia
        findViewById(R.id.btnOpenCv).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mPermissionReady) { //warunek, czy nadano uprawnienia
                    startActivity(new Intent(MainActivity.this, OpenCvRecognizeActivity.class)); //zmiana widoku
                }
            }
        });

        int cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA); //jesli nadano: 0, jesli nie: -1
        int storagePermssion = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);//jesli nadano: 0, jesli nie: -1
        mPermissionReady = cameraPermission == PackageManager.PERMISSION_GRANTED
                && storagePermssion == PackageManager.PERMISSION_GRANTED; //gdy oba uprawnienia nadane, wartosc mPermissionReady: true
        if (!mPermissionReady)
            requirePermissions(); //jesli nie nadano uprawnien, prosba o nadanie
    }

    private void requirePermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, 11); //zapytanie o uprawnienia, request code
    }
    //komunikat o nadaniu uprawnien
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        Map<String, Integer> perm = new HashMap<>();
        perm.put(Manifest.permission.CAMERA, PackageManager.PERMISSION_DENIED);
        perm.put(Manifest.permission.WRITE_EXTERNAL_STORAGE, PackageManager.PERMISSION_DENIED);
        for (int i = 0; i < permissions.length; i++) {
            perm.put(permissions[i], grantResults[i]);
        }
        if (perm.get(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                && perm.get(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            mPermissionReady = true;
        } else {
            if (!ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)
                    || !ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                new AlertDialog.Builder(this)
                        .setMessage(R.string.permission_warning)
                        .setPositiveButton(R.string.dismiss, null)
                        .show();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}