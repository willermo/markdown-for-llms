#!/usr/bin/env python3
"""
Batch conversion monitoring and management tool
"""

import os
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class BatchConversionMonitor:
    """Monitor and manage batch PDF conversions"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.status_file = Path("conversion_status.json")
        
    def is_api_healthy(self) -> bool:
        """Check if the API is responding"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def is_conversion_active(self) -> bool:
        """Check if a conversion is currently running by examining Docker logs"""
        try:
            result = subprocess.run(
                ['docker', 'compose', 'logs', 'marker-api', '--tail', '10'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return False
                
            logs = result.stdout
            
            # Look for active conversion indicators
            active_indicators = [
                "Converting PDF:",
                "Recognizing layout:",
                "Running OCR",
                "Detecting bboxes:",
                "Processing page",
                "%|"  # Progress bar indicator
            ]
            
            return any(indicator in logs for indicator in active_indicators)
            
        except Exception as e:
            print(f"Error checking conversion status: {e}")
            return False
    
    def get_conversion_progress(self) -> Optional[str]:
        """Extract conversion progress from Docker logs"""
        try:
            result = subprocess.run(
                ['docker', 'compose', 'logs', 'marker-api', '--tail', '20'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return None
                
            logs = result.stdout
            lines = logs.split('\n')
            
            # Find the most recent progress line
            for line in reversed(lines):
                if '%|' in line and ('█' in line or '▋' in line):
                    # Extract just the progress part
                    if 'Recognizing layout:' in line:
                        return f"📄 {line.split('Recognizing layout:')[1].strip()}"
                    elif 'Running OCR' in line:
                        return f"🔍 {line.split('Running OCR')[1].strip()}"
                    elif 'Detecting bboxes:' in line:
                        return f"📦 {line.split('Detecting bboxes:')[1].strip()}"
                    else:
                        return f"⚙️ {line.strip()}"
            
            # Look for other status indicators
            for line in reversed(lines):
                if 'Converting PDF:' in line:
                    filename = line.split('Converting PDF:')[1].strip()
                    return f"🔄 Converting: {filename}"
                elif 'completed for' in line:
                    return "✅ Conversion completed"
                elif 'Loading ML models' in line:
                    return "🤖 Loading models..."
                    
            return None
            
        except Exception as e:
            return f"❌ Error reading logs: {e}"
    
    def save_status(self, status: Dict):
        """Save current status to file"""
        status['timestamp'] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def load_status(self) -> Dict:
        """Load status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def monitor_status(self):
        """Display current conversion status"""
        print("🔍 Batch Conversion Monitor")
        print("=" * 50)
        
        # Check API health
        api_healthy = self.is_api_healthy()
        print(f"API Status: {'🟢 Healthy' if api_healthy else '🔴 Unavailable'}")
        
        if not api_healthy:
            print("❌ Marker API is not responding")
            print("💡 Try: docker compose up -d")
            return
        
        # Check for active conversion
        conversion_active = self.is_conversion_active()
        print(f"Conversion Active: {'🟢 Yes' if conversion_active else '🔴 No'}")
        
        if conversion_active:
            progress = self.get_conversion_progress()
            if progress:
                print(f"Progress: {progress}")
            
            print("\n📊 To monitor in real-time:")
            print("docker compose logs marker-api --follow")
            
        else:
            print("✅ No active conversions")
            
        # Show recent status
        status = self.load_status()
        if status:
            print(f"\nLast update: {status.get('timestamp', 'Unknown')}")
    
    def wait_for_completion(self, check_interval: int = 30):
        """Wait for current conversion to complete"""
        if not self.is_conversion_active():
            print("✅ No active conversion detected")
            return True
            
        print("⏳ Waiting for conversion to complete...")
        print(f"📊 Checking every {check_interval} seconds")
        print("📋 Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                if not self.is_conversion_active():
                    print("\n✅ Conversion completed!")
                    return True
                    
                progress = self.get_conversion_progress()
                if progress:
                    print(f"\r{progress}", end="", flush=True)
                else:
                    print(f"\r⏳ Converting... ({datetime.now().strftime('%H:%M:%S')})", end="", flush=True)
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
            return False
    
    def check_queue_safety(self) -> bool:
        """Check if it's safe to start a new conversion"""
        if not self.is_api_healthy():
            print("❌ API not healthy - cannot start conversion")
            return False
            
        if self.is_conversion_active():
            print("⚠️ Conversion already in progress")
            print("📊 Current progress:")
            progress = self.get_conversion_progress()
            if progress:
                print(f"   {progress}")
            return False
            
        print("✅ Safe to start new conversion")
        return True

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor batch PDF conversions")
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--wait', action='store_true', help='Wait for completion')
    parser.add_argument('--check-queue', action='store_true', help='Check if safe to start new conversion')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    monitor = BatchConversionMonitor()
    
    if args.status or (not args.wait and not args.check_queue):
        monitor.monitor_status()
    elif args.wait:
        monitor.wait_for_completion(args.interval)
    elif args.check_queue:
        safe = monitor.check_queue_safety()
        exit(0 if safe else 1)

if __name__ == "__main__":
    main()
