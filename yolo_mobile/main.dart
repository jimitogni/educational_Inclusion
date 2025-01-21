import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'object_detector.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ObjectRecognitionScreen(),
    );
  }
}

class ObjectRecognitionScreen extends StatefulWidget {
  @override
  _ObjectRecognitionScreenState createState() => _ObjectRecognitionScreenState();
}

class _ObjectRecognitionScreenState extends State<ObjectRecognitionScreen> {
  final ObjectDetector _detector = ObjectDetector();
  final ImagePicker _picker = ImagePicker();
  List<String> _detectedObjects = [];
  String _status = "Nenhuma imagem processada.";

  @override
  void initState() {
    super.initState();
    _detector.loadModel();
  }

  Future<void> _pickAndProcessImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      final objects = _detector.detectObjects(File(pickedFile.path));
      setState(() {
        _detectedObjects = objects.cast<String>();
        _status = "Objetos detectados!";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Reconhecimento de Objetos")),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(_status, style: TextStyle(fontSize: 20)),
          ElevatedButton(
            onPressed: _pickAndProcessImage,
            child: Text("Tirar Foto"),
          ),
          if (_detectedObjects.isNotEmpty)
            ..._detectedObjects.map((e) => Text(e)).toList(),
        ],
      ),
    );
  }
}

void predict(File imageFile) async {
  ObjectDetector detector = ObjectDetector();
  await detector.loadModel();
  List<dynamic> results = await detector.detectObjects(imageFile);
  print(results); // Mostre os objetos detectados
}
