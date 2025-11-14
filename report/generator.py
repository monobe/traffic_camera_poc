"""
Report generation module

Generate PDF reports for police and local government
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image
)


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate PDF reports from traffic data"""

    def __init__(
        self,
        storage_manager,
        location: str = "横浜市神奈川区・三ツ沢第25号線",
        recipient: str = "神奈川警察署 交通課"
    ):
        """
        Initialize report generator

        Args:
            storage_manager: StorageManager instance
            location: Location description
            recipient: Report recipient
        """
        self.storage = storage_manager
        self.location = location
        self.recipient = recipient

        logger.info("ReportGenerator initialized")

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
        speed_limit_kmh: float = 30.0
    ):
        """
        Generate PDF report for date range

        Args:
            start_date: Start date
            end_date: End date
            output_path: Output PDF file path
            speed_limit_kmh: Speed limit for reporting
        """
        logger.info(f"Generating report: {start_date.date()} to {end_date.date()}")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Get data
        detections = self.storage.get_detections(start_date, end_date)

        if detections.empty:
            logger.warning("No data found for date range")
            return

        # Calculate statistics
        stats = self._calculate_statistics(detections, speed_limit_kmh)

        # Generate charts
        chart_files = self._generate_charts(detections, speed_limit_kmh)

        # Build PDF
        self._build_pdf(
            output_path,
            start_date,
            end_date,
            stats,
            chart_files,
            speed_limit_kmh
        )

        # Cleanup chart files
        for chart_file in chart_files.values():
            if Path(chart_file).exists():
                Path(chart_file).unlink()

        logger.info(f"✓ Report generated: {output_path}")

    def _calculate_statistics(
        self,
        detections: pd.DataFrame,
        speed_limit_kmh: float
    ) -> dict:
        """Calculate summary statistics from detections"""
        stats = {}

        # Basic counts
        stats['total_vehicles'] = len(detections)
        stats['unique_tracks'] = detections['track_id'].nunique()

        # Speed statistics
        stats['avg_speed'] = detections['speed_kmh'].mean()
        stats['max_speed'] = detections['speed_kmh'].max()
        stats['min_speed'] = detections['speed_kmh'].min()
        stats['median_speed'] = detections['speed_kmh'].median()

        # Speeding
        speeding = detections[detections['speed_kmh'] > speed_limit_kmh]
        stats['speeding_count'] = len(speeding)
        stats['speeding_rate'] = (stats['speeding_count'] / stats['total_vehicles']) * 100

        # By vehicle type
        stats['by_type'] = detections.groupby('object_type').agg({
            'track_id': 'count',
            'speed_kmh': 'mean'
        }).to_dict()

        # By direction
        stats['by_direction'] = detections.groupby('direction').size().to_dict()

        # Peak times
        detections['hour'] = pd.to_datetime(detections['timestamp']).dt.hour
        hourly_counts = detections.groupby('hour').size()
        stats['peak_hour'] = hourly_counts.idxmax()
        stats['peak_count'] = hourly_counts.max()

        return stats

    def _generate_charts(
        self,
        detections: pd.DataFrame,
        speed_limit_kmh: float
    ) -> dict:
        """Generate charts and return file paths"""
        chart_files = {}

        # 1. Speed histogram
        plt.figure(figsize=(10, 6))
        plt.hist(detections['speed_kmh'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(speed_limit_kmh, color='red', linestyle='--', linewidth=2, label=f'制限速度 {speed_limit_kmh}km/h')
        plt.xlabel('速度 (km/h)', fontsize=12)
        plt.ylabel('車両数', fontsize=12)
        plt.title('速度分布', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        chart_files['speed_histogram'] = '/tmp/speed_histogram.png'
        plt.savefig(chart_files['speed_histogram'], dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Hourly traffic volume
        detections['hour'] = pd.to_datetime(detections['timestamp']).dt.hour
        hourly_counts = detections.groupby('hour').size()

        plt.figure(figsize=(12, 6))
        plt.bar(hourly_counts.index, hourly_counts.values, edgecolor='black', alpha=0.7)
        plt.xlabel('時刻', fontsize=12)
        plt.ylabel('車両数', fontsize=12)
        plt.title('時間帯別交通量', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3, axis='y')
        chart_files['hourly_volume'] = '/tmp/hourly_volume.png'
        plt.savefig(chart_files['hourly_volume'], dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Speeding rate by hour
        detections['speeding'] = detections['speed_kmh'] > speed_limit_kmh
        speeding_by_hour = detections.groupby('hour')['speeding'].apply(
            lambda x: (x.sum() / len(x)) * 100 if len(x) > 0 else 0
        )

        plt.figure(figsize=(12, 6))
        plt.plot(speeding_by_hour.index, speeding_by_hour.values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('時刻', fontsize=12)
        plt.ylabel('速度超過率 (%)', fontsize=12)
        plt.title('時間帯別速度超過率', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        chart_files['speeding_rate'] = '/tmp/speeding_rate.png'
        plt.savefig(chart_files['speeding_rate'], dpi=150, bbox_inches='tight')
        plt.close()

        return chart_files

    def _build_pdf(
        self,
        output_path: str,
        start_date: datetime,
        end_date: datetime,
        stats: dict,
        chart_files: dict,
        speed_limit_kmh: float
    ):
        """Build PDF document"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=20*mm,
            bottomMargin=20*mm,
            leftMargin=20*mm,
            rightMargin=20*mm
        )

        # Build story (content)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=10,
            spaceBefore=10
        )

        # Header
        story.append(Paragraph(f"{self.recipient} 御中", styles['Normal']))
        story.append(Spacer(1, 5*mm))

        # Title
        title = f"{self.location}における速度超過及び危険走行に関する調査報告書"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 10*mm))

        # 1. Survey Overview
        story.append(Paragraph("1. 調査概要", heading_style))

        overview_data = [
            ["調査期間", f"{start_date.strftime('%Y年%m月%d日')} ～ {end_date.strftime('%Y年%m月%d日')}"],
            ["調査地点", self.location],
            ["調査方式", "AIカメラによる自動速度推定"],
            ["制限速度", f"{speed_limit_kmh} km/h"]
        ]

        overview_table = Table(overview_data, colWidths=[40*mm, 120*mm])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'HeiseiMin-W3'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))

        story.append(overview_table)
        story.append(Spacer(1, 10*mm))

        # 2. Key Findings
        story.append(Paragraph("2. 主な結果", heading_style))

        findings_data = [
            ["総観測車両数", f"{stats['total_vehicles']:,} 台"],
            ["平均速度", f"{stats['avg_speed']:.1f} km/h"],
            ["最高速度", f"{stats['max_speed']:.1f} km/h"],
            ["速度超過車両数", f"{stats['speeding_count']:,} 台"],
            ["速度超過率", f"{stats['speeding_rate']:.1f} %"],
            ["交通量ピーク時間帯", f"{stats['peak_hour']}時台 ({stats['peak_count']}台)"]
        ]

        findings_table = Table(findings_data, colWidths=[60*mm, 100*mm])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, -1), 'HeiseiMin-W3'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))

        story.append(findings_table)
        story.append(Spacer(1, 10*mm))

        # 3. Charts
        story.append(Paragraph("3. 図表", heading_style))

        # Speed histogram
        if 'speed_histogram' in chart_files:
            story.append(Paragraph("3.1 速度分布", styles['Heading2']))
            img = Image(chart_files['speed_histogram'], width=160*mm, height=96*mm)
            story.append(img)
            story.append(Spacer(1, 5*mm))

        # Hourly volume
        if 'hourly_volume' in chart_files:
            story.append(Paragraph("3.2 時間帯別交通量", styles['Heading2']))
            img = Image(chart_files['hourly_volume'], width=160*mm, height=80*mm)
            story.append(img)
            story.append(Spacer(1, 5*mm))

        # Page break before recommendations
        story.append(PageBreak())

        # Speeding rate
        if 'speeding_rate' in chart_files:
            story.append(Paragraph("3.3 時間帯別速度超過率", styles['Heading2']))
            img = Image(chart_files['speeding_rate'], width=160*mm, height=80*mm)
            story.append(img)
            story.append(Spacer(1, 10*mm))

        # 4. Recommendations
        story.append(Paragraph("4. 要望事項", heading_style))

        recommendations = [
            "Zone30の再点検および速度規制の強化",
            "速度抑制策（ハンプ、狭さく等）の検討",
            "通学時間帯のパトロール強化",
            "速度取締りの実施"
        ]

        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"({i}) {rec}", styles['Normal']))
            story.append(Spacer(1, 3*mm))

        story.append(Spacer(1, 10*mm))

        # 5. Appendix
        story.append(Paragraph("5. 付録", heading_style))

        appendix_text = [
            "【計測方法】",
            "AIカメラ（YOLOv8）による車両検知およびトラッキングにより、車両の移動距離と時間から速度を推定。",
            "",
            "【誤差範囲】",
            f"キャリブレーション精度に依存。推定誤差は±5km/h程度。",
            "",
            "【データ保持】",
            "個人情報（ナンバープレート、顔）は記録せず、統計データのみを保存。"
        ]

        for text in appendix_text:
            story.append(Paragraph(text, styles['Normal']))

        story.append(Spacer(1, 15*mm))

        # Footer
        footer_text = f"報告日：{datetime.now().strftime('%Y年%m月%d日')}"
        story.append(Paragraph(footer_text, styles['Normal']))

        # Build PDF
        doc.build(story)

    def generate_summary_csv(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ):
        """
        Generate summary CSV report

        Args:
            start_date: Start date
            end_date: End date
            output_path: Output CSV file path
        """
        detections = self.storage.get_detections(start_date, end_date)

        if detections.empty:
            logger.warning("No data found for date range")
            return

        # Daily summary
        detections['date'] = pd.to_datetime(detections['timestamp']).dt.date
        daily_summary = detections.groupby('date').agg({
            'track_id': 'count',
            'speed_kmh': ['mean', 'max', 'min']
        })

        daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
        daily_summary.to_csv(output_path)

        logger.info(f"✓ Summary CSV generated: {output_path}")
