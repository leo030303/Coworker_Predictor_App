package com.example.tensorflow_test;
          

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;  
import android.view.View;  
import android.widget.AdapterView;  
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.content.res.AssetManager;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class Main extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
            String[] genList = {"Male", "Female"};
            String[] expList = {"None", "Waiter", "Bar"};
            String[] lanList = {"Basic", "Intermediate", "Fluent"};
            String[] monList = {"January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"};
            String[] eduList = {"None", "Secondary", "College"};

            @Override
            protected void onCreate(Bundle savedInstanceState) {
                super.onCreate(savedInstanceState);  
                setContentView(R.layout.activity_main);  
               //Getting the instance of Spinner and applying OnItemSelectedListener on it  
                Spinner genSpin = findViewById(R.id.genderSpinner);
                Spinner expSpin = findViewById(R.id.experienceSpinner);
                Spinner lanSpin = findViewById(R.id.languageSpinner);
                Spinner monSpin = findViewById(R.id.monthSpinner);
                Spinner eduSpin = findViewById(R.id.educationSpinner);
          
                //Creating the ArrayAdapter instance having the country list  
                ArrayAdapter genAdapter = new ArrayAdapter(this,android.R.layout.simple_spinner_item,genList);
                ArrayAdapter expAdapter = new ArrayAdapter(this,android.R.layout.simple_spinner_item,expList);
                ArrayAdapter lanAdapter = new ArrayAdapter(this,android.R.layout.simple_spinner_item,lanList);
                ArrayAdapter monAdapter = new ArrayAdapter(this,android.R.layout.simple_spinner_item,monList);
                ArrayAdapter eduAdapter = new ArrayAdapter(this,android.R.layout.simple_spinner_item,eduList);
                genAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                expAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                lanAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                monAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                eduAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                //Setting the ArrayAdapter data on the Spinner  
                genSpin.setAdapter(genAdapter);
                expSpin.setAdapter(expAdapter);
                lanSpin.setAdapter(lanAdapter);
                monSpin.setAdapter(monAdapter);
                eduSpin.setAdapter(eduAdapter);
            }  
          
            //Performing action onItemSelected and onNothing selected  
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                Toast.makeText(getApplicationContext(),monList[position] , Toast.LENGTH_LONG).show();
            }
            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
                // TODO Auto-generated method stub
            }
            /** Memory-map the model file in Assets. */
            private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
                    throws IOException {
                AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
                FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
                FileChannel fileChannel = inputStream.getChannel();
                long startOffset = fileDescriptor.getStartOffset();
                long declaredLength = fileDescriptor.getDeclaredLength();
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            }
            public void calculateLength(View view) throws IOException {
                Spinner genSpin = findViewById(R.id.genderSpinner);
                Spinner expSpin = findViewById(R.id.experienceSpinner);
                Spinner lanSpin = findViewById(R.id.languageSpinner);
                Spinner monSpin = findViewById(R.id.monthSpinner);
                Spinner eduSpin = findViewById(R.id.educationSpinner);
                EditText ageText = findViewById(R.id.ageInput);
                String gender = genSpin.getSelectedItem().toString();
                String experience = expSpin.getSelectedItem().toString();
                String language = lanSpin.getSelectedItem().toString();
                String month = monSpin.getSelectedItem().toString();
                String education = eduSpin.getSelectedItem().toString();
                String[] inputsList = {month,ageText.getText().toString(),education,language,gender,experience};
                String[] vocab = {"January", "February", "March", "April", "May", "June", "July",
                        "August", "September", "October", "November", "December", "Male", "Female",
                        "Basic", "Intermediate", "Fluent", "Waiter", "Bar", "None",
                        "Secondary", "College"};
                long[] algInput = new long[6];
                for (int n=0;n<inputsList.length;n++) {
                    if (n == 1){
                        algInput[n] = Integer.parseInt(inputsList[n]);
                    }
                    else{
                        for (int i = 0; i < vocab.length; i++) {
                            if (inputsList[n].equals(vocab[i])){
                                algInput[n] = i+1;
                            }
                        }
                    }
                }
                FloatBuffer output = FloatBuffer.allocate(5);
                MappedByteBuffer modelFile = loadModelFile(this.getApplicationContext().getAssets(), "ConvertedModel.tflite");
                try (Interpreter interpreter = new Interpreter(modelFile)) {
                    interpreter.run(algInput, output);
                    TextView result = findViewById(R.id.resultsBox);
                    String resultText = String.format(Locale.ENGLISH, "You'd survive Le Chateau for %.0f days", output.get(0));
                    result.setText(resultText);
                }
            }
        }