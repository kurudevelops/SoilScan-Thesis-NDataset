import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'core/config/app_config.dart';
import 'features/soil_scanner/presentation/scanner_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (_) {}

  if (AppConfig.enableSupabase) {
    await Supabase.initialize(
      url: AppConfig.supabaseUrl,
      anonKey: AppConfig.supabaseAnonKey,
    );
  }

  runApp(TerraScanApp(cameras: cameras));
}

class TerraScanApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const TerraScanApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Terra Scan',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF6F7F4),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF7CB342),
          primary: const Color(0xFF232323),
          secondary: const Color(0xFF7CB342),
          surface: Colors.white,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: Color(0xFF232323),
          elevation: 0,
          scrolledUnderElevation: 1,
        ),
      ),
      home: ScannerScreen(cameras: cameras),
    );
  }
}