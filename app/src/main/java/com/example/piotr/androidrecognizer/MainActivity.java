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
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private boolean mPermissionReady; //zmienna do kontroli nadanych uprawnien
    private RadioGroup usersRG;
    private Button sprawdzBtn;
    private Button dodajBtn;
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        Log.d("Piopr", "dzialaj no");


        setContentView(R.layout.activity_main); //ustawienie widoku
        //obsluga klikniecia
        usersRG = (RadioGroup) findViewById(R.id.user);
        findViewById(R.id.btnOpenCv).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mPermissionReady) { //warunek, czy nadano uprawnienia
                    startActivity(new Intent(MainActivity.this, OpenCvRecognizeActivity.class)); //zmiana widoku
                }
            }
        });



        sprawdzBtn = (Button) findViewById(R.id.sprawdz);

        sprawdzBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String wiadomosc;
                if(usersRG.getCheckedRadioButtonId()!=-1) {
                    RadioButton selected = (RadioButton) findViewById(usersRG.getCheckedRadioButtonId());
                    wiadomosc = selected.getText().toString();
                    wiadomosc += " id: " + selected.getId();
                    makeListOfUsers();

                    Toast.makeText(getBaseContext(), wiadomosc, Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(getBaseContext(), "Nikogo nie wybrano", Toast.LENGTH_SHORT).show();
                }
            }
        });
        makeListOfUsers();


        dodajBtn = (Button) findViewById(R.id.dodaj);
        dodajBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                if(findViewById(R.id.nazwa).getVisibility()==View.INVISIBLE){
//                    findViewById(R.id.nazwa).setVisibility(View.VISIBLE);
//                } else {
//                    findViewById(R.id.nazwa).setVisibility(View.INVISIBLE);
//                }

                File folder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
                EditText userEditText = (EditText) findViewById(R.id.nazwa);
                String nameOfUser = userEditText.getText().toString().toLowerCase();
                if(nameOfUser.isEmpty()){
                    Toast.makeText(getBaseContext(), "Puste pole tekstowe.", Toast.LENGTH_LONG).show();
                } else {
                    //Toast.makeText(getBaseContext(), nameOfUser, Toast.LENGTH_LONG).show();
                    makeNewUser(nameOfUser);

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

    void makeListOfUsers() {
        File folder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] users = folder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        //test widoku
        usersRG = (RadioGroup) findViewById(R.id.user);
        usersRG.removeAllViews();


        for(String t :users){
            RadioButton userRB = new RadioButton(this);
            userRB.setText(t.substring(1,t.length()));
            usersRG.addView(userRB);
            Log.d("Piopr", t.substring(1,t.length()));
        }
        //userRB.setHeight(100);
        }

    void makeNewUser(String username){
        File trainFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] users = trainFolder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        for(int i =0; i<users.length; i++){
            users[i] = users[i].substring(1, users[i].length());
        }

        //sprawdzanie, czy uzytkownik istnieje
        if(Arrays.asList(users).contains(username)){
            Log.d("Piopr", "Podany uzytkownik istnieje");
            Toast.makeText(getBaseContext(), "Podany uzytkownik istnieje", Toast.LENGTH_LONG).show();
        } else {
            username = Integer.toString(users.length+1) + username;
            Log.d("Piopr", username);
            File createdUser = new File(trainFolder, username);
            createdUser.mkdir();
            makeListOfUsers();


        }

        for(String t : users){
            Log.d("Piopr", t);
        }

    }
}