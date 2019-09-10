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
import android.view.WindowManager;
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

    //TODO: przekazywanie id uzytkownika do nastepnego activity, wybranie folderu w ktorym beda przechowywane zdjecia
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        Log.d("Piopr", "dzialaj no");

        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);
        setContentView(R.layout.activity_main); //ustawienie widoku

        int cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA); //jesli nadano: 0, jesli nie: -1
        int storagePermssion = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);//jesli nadano: 0, jesli nie: -1
        mPermissionReady = cameraPermission == PackageManager.PERMISSION_GRANTED
                && storagePermssion == PackageManager.PERMISSION_GRANTED; //gdy oba uprawnienia nadane, wartosc mPermissionReady: true
        if (!mPermissionReady)
            requirePermissions(); //jesli nie nadano uprawnien, prosba o nadanie
        //tworzenie głównego folderu "train folder"
        File folder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        if(!folder.exists()){
        folder.mkdir();}



        //obsluga klikniecia
        usersRG = (RadioGroup) findViewById(R.id.user);
        findViewById(R.id.btnOpenCv).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if(usersRG.getCheckedRadioButtonId()==-1){
                    Toast.makeText(getBaseContext(), "Nie wybrano uzytkownika lub juz nie istnieje.", Toast.LENGTH_LONG).show();

                } else {
                    RadioButton selected = (RadioButton) findViewById(usersRG.getCheckedRadioButtonId());
                    String username = selected.getText().toString();
                    saveIdAndName(username);
                    Log.d("Piopr", "Nazwa usera: " + TrainHelper.CURRENT_USER +" id: " + TrainHelper.CURRENT_IDUSER);
                    //deleteUser(username);

                    if (mPermissionReady) { //warunek, czy nadano uprawnienia
                        startActivity(new Intent(MainActivity.this, OpenCvRecognizeActivity.class)); //zmiana widoku
                    }
                }


            }
        });




        PomocDialog dialog = new PomocDialog();

        Button sprawdzBtn = (Button) findViewById(R.id.sprawdz);
        sprawdzBtn.setEnabled(true);
        sprawdzBtn.setOnClickListener(view -> {
            dialog.show(getFragmentManager(), "Piopr");

        });
        makeListOfUsers();


        Button dodajBtn = (Button) findViewById(R.id.dodaj);
        dodajBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {


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


        Button usunBtn = (Button) findViewById(R.id.usun);
        usunBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if(usersRG.getCheckedRadioButtonId()==-1){
                    Toast.makeText(getBaseContext(), "Nie wybrano uzytkownika lub juz nie istnieje.", Toast.LENGTH_LONG).show();
                    makeListOfUsers();
                } else {
                    RadioButton selected = (RadioButton) findViewById(usersRG.getCheckedRadioButtonId());
                    String username = selected.getText().toString();
                    deleteUser(username);
                }
            }
        });






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

    private void makeListOfUsers() {
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
            userRB.setText(t.replaceAll("[0-9]",""));
            usersRG.addView(userRB);
            Log.d("Piopr", t);
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
        Integer[] usersIds = getUserIds(users);
        users = getUserNames(users);

        String komunikat = "";
        for(int i=0; i<users.length; i++){
            komunikat+=usersIds[i]+" "+users[i]+"\n";
        }
        Log.d("Piopr", komunikat);

        //sprawdzanie, czy uzytkownik istnieje
        if(Arrays.asList(users).contains(username)){
            Log.d("Piopr", "Podany uzytkownik istnieje");
            Toast.makeText(getBaseContext(), "Podany uzytkownik istnieje", Toast.LENGTH_LONG).show();
        } else {

            //szukanie max id dla poprawnego nadawania id uzytkownikom
            int maxId = 0;
            for(int t : usersIds){
                maxId = Math.max(maxId, t);
            }
            username = Integer.toString(maxId+1) + username;
            File createdUser = new File(trainFolder, username);
            createdUser.mkdir();
            File defaultFolder = new File(createdUser, "default");
            defaultFolder.mkdir();
            File eigenFolder = new File(createdUser, "visualizations");
            eigenFolder.mkdir();
            makeListOfUsers();


        }
        Log.d("Piopr", username);
        for(String t : users){
            Log.d("Piopr", t);
        }
    }

    void deleteUser(String username){
        File trainFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] userList = trainFolder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        Integer[] userIds = getUserIds(userList);
        String[] userNames = getUserNames(userList);
        int indexOfFound = Arrays.asList(userNames).indexOf(username);
        Log.d("Piopr", "Znaleziony index uzytkownika: " + indexOfFound);
        Log.d("Piopr", "Czy poprawny id?: " + Arrays.asList(userIds).get(indexOfFound) + " dla " +
                Arrays.asList(userNames).get(indexOfFound));

        String nameOfFolder;
        nameOfFolder = Arrays.asList(userIds).get(indexOfFound).toString() + Arrays.asList(userNames).get(indexOfFound);

        File folderToDelete = new File(trainFolder, nameOfFolder);

        deleteRecursive(folderToDelete);
        makeListOfUsers();



    }

    /**
     * Zwraca id wszystkich uzytkownikow (wyniki pasuja do tablicy z metody getUserNames())
     * @param users - lista folderow uzytkownikow (id + nazwa)
     * @return tablica id uzytkownikow typu int
     */
    Integer[] getUserIds(String[] users){
        Integer[] usersIds = new Integer[users.length];
        for(int i = 0; i<users.length; i++){
            usersIds[i]=Integer.parseInt(users[i].replaceAll("[^0-9]", ""));
        }
        return usersIds;
    }

    String[] getUserNames(String[] users){
        for(int i =0; i<users.length; i++){
            users[i] = users[i].replaceAll("[0-9]","");
        }
        return users;
    }

    /***
     * Rekursywne usuwanie folderów i plikow
     *
     * @param fileOrDirectory - folder, który chcemy usunac (podfoldery takze zostana usuniete)
     */
    void deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory())
            for (File child : fileOrDirectory.listFiles())
                deleteRecursive(child);

        fileOrDirectory.delete();
    }

    void saveIdAndName(String username){
        File trainFolder = new File("/mnt/sdcard/", TrainHelper.TRAIN_FOLDER);
        String[] userList = trainFolder.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        Integer[] userIds = getUserIds(userList);
        String[] userNames = getUserNames(userList);
        int indexOfFound = Arrays.asList(userNames).indexOf(username);

        TrainHelper.CURRENT_IDUSER = Arrays.asList(userIds).get(indexOfFound);
        TrainHelper.CURRENT_USER = Arrays.asList(userNames).get(indexOfFound);
        TrainHelper.CURRENT_FOLDER = TrainHelper.CURRENT_IDUSER+TrainHelper.CURRENT_USER;
        Log.d("Piopr", "Aktualny folder: " + TrainHelper.CURRENT_FOLDER);
    }

}

