import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../../core/config/app_config.dart';
import '../../../core/storage/local_queue_storage.dart';
import '../data/scan_repository.dart';
import '../domain/scan_record.dart';
import 'result_card.dart';

class ScannerScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const ScannerScreen({super.key, required this.cameras});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  CameraController? _controller;
  final ScanRepository _repo = ScanRepository();
  final LocalQueueStorage _storage = LocalQueueStorage();
  List<ScanRecord> _scans = [];
  bool _isBusy = false;

  @override
  void initState() {
    super.initState();
    _loadSavedUrlAndScans();
    _initCamera();
  }

  Future<void> _loadSavedUrlAndScans() async {
    final savedUrl = await _storage.loadApiUrl();
    if (savedUrl != null && savedUrl.isNotEmpty) {
      AppConfig.apiBaseUrl = savedUrl;
    }
    await _refreshList();
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;
    try {
      _controller = CameraController(
        widget.cameras.first,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Future<void> _refreshList() async {
    final list = await _storage.getAllScans();
    if (mounted) setState(() => _scans = list);
  }

  Future<void> _takePhoto() async {
    if (_controller == null || !_controller!.value.isInitialized || _isBusy) return;
    setState(() => _isBusy = true);
    try {
      final xfile = await _controller!.takePicture();
      final bytes = await xfile.readAsBytes();
      await _repo.queueAndAttemptSync(bytes);
      await _refreshList();
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  Future<void> _pickGallery() async {
    final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (picked == null) return;
    setState(() => _isBusy = true);
    try {
      final bytes = await picked.readAsBytes();
      await _repo.queueAndAttemptSync(bytes);
      await _refreshList();
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  Future<void> _syncAll() async {
    setState(() => _isBusy = true);
    try {
      final count = await _repo.syncAllPending();
      await _refreshList();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Synced $count pending scan(s).'),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  void _editServerUrl() {
    final ctrl = TextEditingController(text: AppConfig.apiBaseUrl);
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text(
          'Backend API Connection',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter your Render Cloud URL or Laptop Hotspot IP:',
              style: TextStyle(fontSize: 13, color: Colors.black54),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: ctrl,
              decoration: InputDecoration(
                hintText: 'https://terra-scan-api.onrender.com',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 10,
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: const Color(0xFF7CB342),
            ),
            onPressed: () async {
              final clean = ctrl.text.trim().replaceAll(RegExp(r'/$'), '');
              AppConfig.apiBaseUrl = clean;
              await _storage.saveApiUrl(clean);
              if (mounted) {
                setState(() {});
                Navigator.pop(context);
              }
            },
            child: const Text('Save Endpoint'),
          ),
        ],
      ),
    );
  }

  Widget _buildTrueAspectCamera() {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        _controller!.value.previewSize == null) {
      return Container(
        color: const Color(0xFF1E1E1E),
        alignment: Alignment.center,
        child: const Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.camera_alt_outlined, color: Colors.white38, size: 42),
            SizedBox(height: 8),
            Text(
              'Initializing Camera...',
              style: TextStyle(color: Colors.white60, fontSize: 13),
            ),
          ],
        ),
      );
    }

    // Native sensor previewSize is landscape; swap width & height for portrait display
    final previewSize = _controller!.value.previewSize!;
    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: previewSize.height,
          height: previewSize.width,
          child: CameraPreview(_controller!),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final pendingCount = _scans.where((s) => s.status == 'pending_offline').length;

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 16,
        title: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.asset(
                'assets/terra_scan_logo.png',
                width: 36,
                height: 36,
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => Container(
                  width: 36,
                  height: 36,
                  decoration: const BoxDecoration(
                    color: Color(0xFF232323),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.eco, color: Color(0xFF7CB342), size: 20),
                ),
              ),
            ),
            const SizedBox(width: 12),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'TERRA SCAN',
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 17,
                    letterSpacing: 2.0,
                    color: Color(0xFF232323),
                  ),
                ),
                Text(
                  'A G R I C A P T U R E',
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 9,
                    letterSpacing: 1.5,
                    color: Color(0xFF7CB342),
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.dns_outlined),
            tooltip: 'Configure Server URL',
            onPressed: _editServerUrl,
          ),
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: IconButton(
              icon: Badge(
                backgroundColor: const Color(0xFF7CB342),
                isLabelVisible: pendingCount > 0,
                label: Text('$pendingCount'),
                child: const Icon(Icons.cloud_sync_outlined),
              ),
              tooltip: 'Sync Pending Scans',
              onPressed: _isBusy ? null : _syncAll,
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // 1. Distortion-Free Rounded Viewfinder Card
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: SizedBox(
                height: 260,
                width: double.infinity,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    _buildTrueAspectCamera(),
                    IgnorePointer(
                      child: CustomPaint(
                        painter: _WhiteReferenceOverlayPainter(),
                      ),
                    ),
                    Positioned(
                      top: 12,
                      left: 16,
                      right: 16,
                      child: Center(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.65),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text(
                            'Center soil in circle • Keep white paper in outer frame',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 11,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // 2. Clean Action Control Bar (Separate from camera view)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      side: const BorderSide(color: Color(0xFF232323), width: 1.2),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                    onPressed: _isBusy ? null : _pickGallery,
                    icon: const Icon(Icons.photo_library_outlined, size: 20),
                    label: const Text(
                      'Upload Sample',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: const Color(0xFF7CB342),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                    onPressed: _isBusy ? null : _takePhoto,
                    icon: _isBusy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.camera_alt_outlined, size: 20),
                    label: Text(
                      _isBusy ? 'Processing...' : 'Capture Soil',
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
              ],
            ),
          ),

          // 3. Section Header + Active Endpoint Pill
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 10, 18, 4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'RECENT SOIL ANALYSES',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 1.1,
                    color: Colors.black54,
                  ),
                ),
                GestureDetector(
                  onTap: _editServerUrl,
                  child: Text(
                    AppConfig.apiBaseUrl.replaceFirst(RegExp(r'^https?://'), ''),
                    style: const TextStyle(
                      fontSize: 11,
                      color: Color(0xFF7CB342),
                      fontWeight: FontWeight.w600,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
              ],
            ),
          ),

          // 4. Scan Results List
          Expanded(
            child: _scans.isEmpty
                ? const Center(
                    child: Text(
                      'No soil samples scanned yet.\nCapture or upload a sample above.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.black45, fontSize: 13),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.only(top: 4, bottom: 20),
                    itemCount: _scans.length,
                    itemBuilder: (ctx, i) {
                      final s = _scans[i];
                      return ResultCard(
                        scan: s,
                        onRetrySync: s.status == 'pending_offline'
                            ? () async {
                                setState(() => _isBusy = true);
                                try {
                                  await _repo.syncSingleRecord(s);
                                  await _refreshList();
                                } catch (e) {
                                  if (mounted) {
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text(
                                          'Connection failed (${AppConfig.apiBaseUrl}). Tap the server icon at the top right to set your Render URL or Laptop IP.',
                                        ),
                                        behavior: SnackBarBehavior.floating,
                                      ),
                                    );
                                  }
                                } finally {
                                  if (mounted) setState(() => _isBusy = false);
                                }
                              }
                            : null,
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _WhiteReferenceOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final borderPaint = Paint()
      ..color = Colors.white.withOpacity(0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final soilPaint = Paint()
      ..color = const Color(0xFF7CB342)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final rect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2 + 10),
      width: size.width * 0.80,
      height: size.height * 0.72,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(16)),
      borderPaint,
    );

    canvas.drawCircle(
      Offset(size.width / 2, size.height / 2 + 10),
      size.height * 0.25,
      soilPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}