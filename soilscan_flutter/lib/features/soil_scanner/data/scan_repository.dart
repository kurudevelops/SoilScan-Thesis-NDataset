import 'dart:convert';
import 'dart:typed_data';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

import '../../../core/config/app_config.dart';
import '../../../core/storage/local_queue_storage.dart';
import '../domain/scan_record.dart';

class ScanRepository {
  final LocalQueueStorage _storage = LocalQueueStorage();

  Future<ScanRecord> queueAndAttemptSync(Uint8List imageBytes) async {
    final b64 = base64Encode(imageBytes);
    var record = ScanRecord(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      createdAt: DateTime.now().toIso8601String(),
      status: 'pending_offline',
      rawImageB64: b64,
    );

    await _storage.saveOrUpdateScan(record);

    final connectivity = await Connectivity().checkConnectivity();
    if (connectivity != ConnectivityResult.none) {
      try {
        record = await syncSingleRecord(record);
      } catch (_) {
        // Stays queued offline if server is unreachable
      }
    }
    return record;
  }

  Future<ScanRecord> syncSingleRecord(ScanRecord record) async {
    final bytes = base64Decode(record.rawImageB64);
    final uri = Uri.parse('${AppConfig.apiBaseUrl}/api/v1/scan');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: 'soil_${record.id}.jpg',
        ),
      );

    final streamed = await request.send().timeout(const Duration(seconds: 45));
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception('Server returned ${response.statusCode}: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final results = data['results'] as Map<String, dynamic>;
    final flags = (data['qc_flags'] as List<dynamic>).join('; ');

    final updated = ScanRecord(
      id: record.id,
      createdAt: record.createdAt,
      status: 'synced',
      rawImageB64: record.rawImageB64,
      nitrogen: results['N']['level'] as String,
      phosphorus: results['P']['level'] as String,
      potassium: results['K']['level'] as String,
      nConf: (results['N']['confidence'] as num).toDouble(),
      pConf: (results['P']['confidence'] as num).toDouble(),
      kConf: (results['K']['confidence'] as num).toDouble(),
      pattern: data['pattern'] as String,
      modeDetected: data['mode_detected'] as String,
      qcFlags: flags,
      isolatedPreviewB64: data['isolated_preview_base64'] as String?,
    );

    await _storage.saveOrUpdateScan(updated);

    if (AppConfig.enableSupabase) {
      try {
        await Supabase.instance.client.from('soil_scans').insert({
          'nitrogen': updated.nitrogen,
          'phosphorus': updated.phosphorus,
          'potassium': updated.potassium,
          'pattern': updated.pattern,
          'n_conf': updated.nConf,
          'p_conf': updated.pConf,
          'k_conf': updated.kConf,
          'mode_detected': updated.modeDetected,
        });
      } catch (_) {}
    }

    return updated;
  }

  Future<int> syncAllPending() async {
    final pending = await _storage.getPendingScans();
    int count = 0;
    for (final rec in pending) {
      try {
        await syncSingleRecord(rec);
        count++;
      } catch (_) {}
    }
    return count;
  }
}