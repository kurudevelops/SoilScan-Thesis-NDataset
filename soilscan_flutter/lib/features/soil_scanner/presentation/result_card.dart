import 'dart:convert';
import 'package:flutter/material.dart';
import '../domain/scan_record.dart';

class ResultCard extends StatelessWidget {
  final ScanRecord scan;
  final VoidCallback? onRetrySync;

  const ResultCard({super.key, required this.scan, this.onRetrySync});

  Color _badgeColor(String? level) {
    switch (level) {
      case 'L':
        return const Color(0xFFE65100);
      case 'M':
        return const Color(0xFF1565C0);
      case 'H':
        return const Color(0xFF558B2F);
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    final isSynced = scan.status == 'synced';
    final previewB64 = isSynced && scan.isolatedPreviewB64 != null
        ? scan.isolatedPreviewB64!
        : scan.rawImageB64;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    base64Decode(previewB64),
                    width: 80,
                    height: 80,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: isSynced
                      ? Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 8,
                                    vertical: 2,
                                  ),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF7CB342).withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(6),
                                  ),
                                  child: Text(
                                    'Pattern ${scan.pattern ?? "---"} • ${(scan.modeDetected ?? "paper").toUpperCase()}',
                                    style: const TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.w800,
                                      color: Color(0xFF558B2F),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                Expanded(
                                  child: _nutrientTile('Nitrogen', scan.nitrogen, scan.nConf),
                                ),
                                const SizedBox(width: 6),
                                Expanded(
                                  child: _nutrientTile('Phosphorus', scan.phosphorus, scan.pConf),
                                ),
                                const SizedBox(width: 6),
                                Expanded(
                                  child: _nutrientTile('Potassium', scan.potassium, scan.kConf),
                                ),
                              ],
                            ),
                          ],
                        )
                      : Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  width: 8,
                                  height: 8,
                                  decoration: const BoxDecoration(
                                    color: Colors.deepOrange,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 6),
                                const Text(
                                  'Saved Offline (Queued)',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w800,
                                    fontSize: 13,
                                    color: Colors.deepOrange,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              'Ready for White Balance + HSV Lasso + 3-Model Stack analysis.',
                              style: TextStyle(fontSize: 12, color: Colors.black54),
                            ),
                            const SizedBox(height: 6),
                            if (onRetrySync != null)
                              SizedBox(
                                height: 32,
                                child: FilledButton.tonalIcon(
                                  style: FilledButton.styleFrom(
                                    backgroundColor: const Color(0xFF7CB342).withOpacity(0.15),
                                    foregroundColor: const Color(0xFF33691E),
                                    padding: const EdgeInsets.symmetric(horizontal: 12),
                                  ),
                                  onPressed: onRetrySync,
                                  icon: const Icon(Icons.sync, size: 15),
                                  label: const Text(
                                    'Analyze Now',
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                ),
              ],
            ),
            if (isSynced && scan.qcFlags != null && scan.qcFlags!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.amber.shade50,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.amber.shade300),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline, size: 15, color: Colors.orange),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          'Calibration Note: ${scan.qcFlags}',
                          style: const TextStyle(fontSize: 11, color: Colors.black87),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _nutrientTile(String title, String? level, double? conf) {
    final pct = conf != null ? '${(conf * 100).toStringAsFixed(0)}%' : '';
    final color = _badgeColor(level);
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: Colors.black54,
            ),
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 2),
          Text(
            level ?? '-',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w900,
              color: color,
            ),
          ),
          if (pct.isNotEmpty)
            Text(
              pct,
              style: TextStyle(
                fontSize: 9,
                fontWeight: FontWeight.w600,
                color: color.withOpacity(0.8),
              ),
            ),
        ],
      ),
    );
  }
}