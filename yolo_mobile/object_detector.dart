import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';

class ObjectDetector {
  late Interpreter interpreter;

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('model.tflite');
  }

  List<dynamic> detectObjects(File imageFile) {
    // Aqui você pode implementar o processamento da imagem e a detecção.
    // Por enquanto, retornaremos um placeholder.
    return ['Objeto 1', 'Objeto 2'];
  }
}

import 'package:flutter_tts/flutter_tts.dart';

class _ObjectRecognitionScreenState extends State<ObjectRecognitionScreen> {
  final FlutterTts _flutterTts = FlutterTts();

  Future<void> _speak(String text) async {
    await _flutterTts.speak(text);
  }

  Future<void> _pickAndProcessImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      final objects = _detector.detectObjects(File(pickedFile.path));
      setState(() {
        _detectedObjects = objects.cast<String>();
        _status = "Objetos detectados!";
      });
      _speak("Objetos detectados: ${_detectedObjects.join(', ')}");
    }
  }
}

import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'dart:io';
import 'package:image/image.dart' as img;

class ObjectDetector {
  late Interpreter interpreter;

  Future<void> loadModel() async {
    // Carregue o modelo TFLite
    interpreter = await Interpreter.fromAsset('assets/models/model.tflite');
  }

  Future<List<dynamic>> detectObjects(File imageFile) async {
    // Transforme a imagem em uma matriz adequada para o modelo
    final image = img.decodeImage(imageFile.readAsBytesSync());
    final resizedImage = img.copyResize(image!, width: 224, height: 224);

    // Converta a imagem para um tensor
    final input = _imageToByteList(resizedImage);
    final output = List.generate(1, (index) => List.filled(100, 0)); // Ajuste conforme necessário

    // Faça a predição
    interpreter.run(input, output);

    return output;
  }

  Uint8List _imageToByteList(img.Image image) {
    final convertedBytes = Uint8List(1 * 224 * 224 * 3);
    var buffer = ByteData.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = image.getPixel(x, y);
        buffer.setFloat32(pixelIndex++, img.getRed(pixel) / 255.0);
        buffer.setFloat32(pixelIndex++, img.getGreen(pixel) / 255.0);
        buffer.setFloat32(pixelIndex++, img.getBlue(pixel) / 255.0);
      }
    }
    return convertedBytes;
  }
}

