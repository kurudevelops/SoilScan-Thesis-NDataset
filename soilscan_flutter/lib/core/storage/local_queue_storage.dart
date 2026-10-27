import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../../features/soil_scanner/domain/scan_record.dart';

class LocalQueueStorage {
  static const String _storageKey = 'soilscan_offline_records_v1';
  static const String _apiKey = 'soilscan_api_base_url';

  Future<void> saveApiUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_apiKey, url);
  }

  Future<String?> loadApiUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_apiKey);
  }

  Future<List<ScanRecord>> getAllScans() async {
    final prefs = await SharedPreferences.getInstance();
    final rawList = prefs.getStringList(_storageKey) ?? [];
    return rawList
        .map((item) => ScanRecord.fromJson(jsonDecode(item) as Map<String, dynamic>))
        .toList();
  }

  Future<void> saveOrUpdateScan(ScanRecord record) async {
    final prefs = await SharedPreferences.getInstance();
    final current = await getAllScans();
    final idx = current.indexWhere((r) => r.id == record.id);
    if (idx >= 0) {
      current[idx] = record;
    } else {
      current.insert(0, record);
    }
    final encoded = current.map((r) => jsonEncode(r.toJson())).toList();
    await prefs.setStringList(_storageKey, encoded);
  }

  Future<List<ScanRecord>> getPendingScans() async {
    final all = await getAllScans();
    return all.where((r) => r.status == 'pending_offline').toList();
  }

  Future<void> clearAll() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }
}