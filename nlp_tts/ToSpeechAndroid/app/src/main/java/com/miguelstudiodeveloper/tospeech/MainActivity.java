package com.miguelstudiodeveloper.tospeech;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.util.Locale;

public class MainActivity extends AppCompatActivity implements TextToSpeech.OnInitListener {

    private Button reproduzir;
    private EditText texto;
    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //init UI components
        reproduzir = findViewById(R.id.reproduzir);
        texto = findViewById(R.id.texto);

        //init text to speech
        tts = new TextToSpeech(this, this);

        //set onclick
        reproduzir.setOnClickListener(new Escutador());

    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS){
            int result = tts.setLanguage(Locale.getDefault());
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED){
                Toast.makeText(getApplicationContext(), "Idioma não suportado...", Toast.LENGTH_SHORT).show();
            }else{
                falarTexto("Compatibilidade ok!");
            }
        }
    }

    private class Escutador implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            String t = texto.getText().toString();
            falarTexto(t);
            texto.setText("");
            texto.setHint("Texto a ser falado...");
        }
    }

    private void falarTexto(String t) {
        tts.speak(t, TextToSpeech.QUEUE_FLUSH, null);
    }
}