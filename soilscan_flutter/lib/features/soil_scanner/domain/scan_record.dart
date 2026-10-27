class ScanRecord {
  final String id;
  final String createdAt;
  final String status; // 'pending_offline' or 'synced'
  final String rawImageB64;
  final String? nitrogen; // 'L', 'M', 'H'
  final String? phosphorus;
  final String? potassium;
  final double? nConf;
  final double? pConf;
  final double? kConf;
  final String? pattern;
  final String? modeDetected;
  final String? qcFlags;
  final String? isolatedPreviewB64;

  ScanRecord({
    required this.id,
    required this.createdAt,
    required this.status,
    required this.rawImageB64,
    this.nitrogen,
    this.phosphorus,
    this.potassium,
    this.nConf,
    this.pConf,
    this.kConf,
    this.pattern,
    this.modeDetected,
    this.qcFlags,
    this.isolatedPreviewB64,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'createdAt': createdAt,
        'status': status,
        'rawImageB64': rawImageB64,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'nConf': nConf,
        'pConf': pConf,
        'kConf': kConf,
        'pattern': pattern,
        'modeDetected': modeDetected,
        'qcFlags': qcFlags,
        'isolatedPreviewB64': isolatedPreviewB64,
      };

  factory ScanRecord.fromJson(Map<String, dynamic> json) => ScanRecord(
        id: json['id'] as String,
        createdAt: json['createdAt'] as String,
        status: json['status'] as String,
        rawImageB64: json['rawImageB64'] as String,
        nitrogen: json['nitrogen'] as String?,
        phosphorus: json['phosphorus'] as String?,
        potassium: json['potassium'] as String?,
        nConf: (json['nConf'] as num?)?.toDouble(),
        pConf: (json['pConf'] as num?)?.toDouble(),
        kConf: (json['kConf'] as num?)?.toDouble(),
        pattern: json['pattern'] as String?,
        modeDetected: json['modeDetected'] as String?,
        qcFlags: json['qcFlags'] as String?,
        isolatedPreviewB64: json['isolatedPreviewB64'] as String?,
      );
}