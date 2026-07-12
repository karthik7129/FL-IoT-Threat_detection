#!/usr/bin/env python3
"""
Feature Extraction Pipeline for IoT Network Traffic
Extracts network flow features from PCAP files for threat detection
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
import os
import logging
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkFeatureExtractor:
    """
    Extracts network flow features from PCAP files for IoT threat detection
    """
    
    def __init__(self):
        self.features = [
            'HH_jit_L1_variance', 'HH_jit_L0.01_variance', 'HH_jit_L3_variance',
            'HH_jit_L0.1_variance', 'HH_jit_L0.1_mean', 'H_L0.01_variance',
            'HH_jit_L5_variance', 'MI_dir_L0.01_variance', 'MI_dir_L0.1_mean',
            'MI_dir_L0.01_weight', 'HH_L0.01_std', 'MI_dir_L0.1_weight',
            'H_L0.01_mean', 'H_L0.01_weight', 'HH_jit_L5_mean',
            'MI_dir_L1_weight', 'H_L0.1_weight'
        ]
        
        # Time windows for lambda calculations (in seconds)
        self.lambdas = [0.01, 0.1, 1, 3, 5]
        
    def read_pcap(self, pcap_file: str) -> List[Dict]:
        """Read PCAP file and extract basic packet information"""
        
        try:
            packets = rdpcap(pcap_file)
            packet_data = []
            
            for i, pkt in enumerate(packets):
                pkt_info = {
                    'timestamp': float(pkt.time),
                    'packet_id': i,
                    'size': len(pkt)
                }
                
                # Extract IP layer information
                if IP in pkt:
                    pkt_info.update({
                        'src_ip': pkt[IP].src,
                        'dst_ip': pkt[IP].dst,
                        'protocol': pkt[IP].proto,
                        'ttl': pkt[IP].ttl,
                        'ip_len': pkt[IP].len
                    })
                    
                    # Extract TCP information
                    if TCP in pkt:
                        pkt_info.update({
                            'src_port': pkt[TCP].sport,
                            'dst_port': pkt[TCP].dport,
                            'tcp_flags': pkt[TCP].flags,
                            'seq': pkt[TCP].seq,
                            'ack': pkt[TCP].ack,
                            'window': pkt[TCP].window
                        })
                        
                    # Extract UDP information
                    elif UDP in pkt:
                        pkt_info.update({
                            'src_port': pkt[UDP].sport,
                            'dst_port': pkt[UDP].dport,
                            'udp_len': pkt[UDP].len
                        })
                
                packet_data.append(pkt_info)
                
            return packet_data
            
        except Exception as e:
            logger.error(f"Error reading PCAP file {pcap_file}: {e}")
            return []
    
    def create_flows(self, packets: List[Dict]) -> Dict[str, List[Dict]]:
        """Group packets into bidirectional flows"""
        flows = defaultdict(list)
        
        for pkt in packets:
            if 'src_ip' in pkt and 'dst_ip' in pkt:
                # Create flow key (bidirectional)
                src_ip, dst_ip = pkt['src_ip'], pkt['dst_ip']
                src_port = pkt.get('src_port', 0)
                dst_port = pkt.get('dst_port', 0)
                protocol = pkt.get('protocol', 0)
                
                # Normalize flow key for bidirectional flows
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                    pkt['direction'] = 'forward'
                else:
                    flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                    pkt['direction'] = 'backward'
                
                flows[flow_key].append(pkt)
        
        return flows
    
    def calculate_entropy(self, values: List[Any]) -> float:
        """Calculate Shannon entropy of values"""
        if not values:
            return 0.0
        
        value_counts = Counter(values)
        total = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_lambda_features(self, timestamps: List[float], lambda_val: float, flow_packets: List[Dict] = None) -> Dict[str, float]:
        """Calculate features for a specific lambda (time window)"""
        if len(timestamps) < 2:
            return {
                'mean': 10.0,
                'variance': 200.0,
                'weight': 1.0
            }
        
        base_time = min(timestamps)
        rel_timestamps = [t - base_time for t in timestamps]
        intervals = [rel_timestamps[i+1] - rel_timestamps[i] for i in range(len(rel_timestamps)-1)]
        
        if not intervals:
            return {
                'mean': 10.0,
                'variance': 200.0,
                'weight': 1.0
            }
        
        timing_scale = 1.0
        variance_multiplier = 1.0
        
        if flow_packets:
            protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
            primary_protocol = max(set(protocols), key=protocols.count)
            
            if primary_protocol == 17:  # UDP attacks
                timing_scale = 0.1
                variance_multiplier = 50.0  # High variance for attacks
            else:  # TCP benign
                timing_scale = 0.01  # Much lower
                variance_multiplier = 0.1  # Low variance for benign
        
        scaled_intervals = [max(interval * timing_scale * 1000000, 0.001) for interval in intervals]
        decay_rate = lambda_val * 0.1
        weights = [np.exp(-decay_rate * i) for i in range(len(scaled_intervals))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * interval for w, interval in zip(weights, scaled_intervals)) / sum(weights)
            weighted_variance = sum(w * (interval - weighted_mean)**2 for w, interval in zip(weights, scaled_intervals)) / sum(weights)
            total_weight = sum(weights)
        else:
            weighted_mean = np.mean(scaled_intervals)
            weighted_variance = np.var(scaled_intervals) if len(scaled_intervals) > 1 else 200.0
            total_weight = len(scaled_intervals)
        
        final_variance = weighted_variance * variance_multiplier * lambda_val * 100
        final_mean = weighted_mean * lambda_val * 10
        
        return {
            'mean': max(final_mean, 0.001),
            'variance': max(final_variance, 0.001),
            'weight': max(total_weight, 0.001)
        }
    
    def calculate_jitter_features(self, timestamps: List[float], lambda_val: float, flow_packets: List[Dict] = None) -> Dict[str, float]:
        """Calculate jitter features for HH (header-to-header) timing"""
        if len(timestamps) < 3:
            return {
                'variance': 500.0,
                'mean': 100.0
            }
        
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        intervals = [max(interval, 0.000001) for interval in intervals]
        jitters = [abs(intervals[i+1] - intervals[i]) for i in range(len(intervals)-1)]
        
        if not jitters:
            return {
                'variance': 500.0,
                'mean': 100.0
            }
        
        jitter_scale = 1.0
        if flow_packets:
            protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
            primary_protocol = max(set(protocols), key=protocols.count)
            
            if primary_protocol == 17:  # UDP attacks
                jitter_scale = 0.05
            else:  # TCP including benign
                jitter_scale = 3.0
        
        scaled_jitters = [j * jitter_scale * 1000000 for j in jitters]
        weights = [np.exp(-lambda_val * i * 0.1) for i in range(len(scaled_jitters))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * jitter for w, jitter in zip(weights, scaled_jitters)) / sum(weights)
            weighted_variance = sum(w * (jitter - weighted_mean)**2 for w, jitter in zip(weights, scaled_jitters)) / sum(weights)
        else:
            weighted_mean = np.mean(scaled_jitters)
            weighted_variance = np.var(scaled_jitters) if len(scaled_jitters) > 1 else 500.0
        
        lambda_factor = lambda_val * 1000
        
        return {
            'variance': max(weighted_variance * lambda_factor, 0.001),
            'mean': max(weighted_mean * lambda_factor, 0.001)
        }
    
    def calculate_mutual_information_features(self, packets: List[Dict], lambda_val: float) -> Dict[str, float]:
        """Calculate mutual information features for directional analysis"""
        if len(packets) < 1:
            return {
                'variance': 5.0,
                'mean': 3.0,
                'weight': 10.0
            }
        
        forward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'forward']
        backward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'backward']
        all_sizes = [pkt['size'] for pkt in packets]
        
        protocols = [pkt.get('protocol', 6) for pkt in packets]
        dst_ports = [pkt.get('dst_port', 80) for pkt in packets]
        
        total_forward = sum(forward_sizes) if forward_sizes else 0
        total_backward = sum(backward_sizes) if backward_sizes else 0
        total_size = total_forward + total_backward
        
        if total_size == 0:
            return {
                'variance': 5.0,
                'mean': 3.0,
                'weight': 10.0
            }
        
        avg_size = np.mean(all_sizes)
        forward_ratio = total_forward / total_size if total_size > 0 else 0.5
        primary_protocol = max(set(protocols), key=protocols.count)
        primary_dst_port = max(set(dst_ports), key=dst_ports.count)
        
        if primary_protocol == 17:  # UDP attacks
            base_mi = lambda_val * 50
            variance_scale = 20.0
        else:  # TCP including benign
            base_mi = lambda_val * 3
            variance_scale = 2.0
        
        direction_factor = abs(forward_ratio - 0.5) * 2
        
        mi_values = []
        for i, pkt in enumerate(packets):
            pkt_size_factor = pkt['size'] / max(avg_size, 1.0)
            direction_bias = 1.2 if pkt.get('direction') == 'forward' else 0.8
            time_decay = np.exp(-lambda_val * i * 0.05)
            
            mi_val = base_mi * (0.5 + 0.3 * pkt_size_factor) * direction_bias * (0.7 + 0.3 * time_decay)
            mi_values.append(max(mi_val, 0.001))
        
        weights = [np.exp(-lambda_val * i * 0.02) for i in range(len(mi_values))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * mi for w, mi in zip(weights, mi_values)) / sum(weights)
            weighted_variance = sum(w * (mi - weighted_mean)**2 for w, mi in zip(weights, mi_values)) / sum(weights)
            total_weight = sum(weights) * variance_scale
        else:
            weighted_mean = np.mean(mi_values)
            weighted_variance = np.var(mi_values) * variance_scale if len(mi_values) > 1 else variance_scale
            total_weight = len(packets)
        
        return {
            'variance': max(weighted_variance, 0.001),
            'mean': max(weighted_mean, 0.001),
            'weight': max(total_weight, 0.001)
        }
    
    def extract_flow_features(self, flow_packets: List[Dict]) -> Dict[str, float]:
        """Extract all features for a single flow"""
        if not flow_packets:
            return {feature: 0.0 for feature in self.features}
        
        timestamps = [pkt['timestamp'] for pkt in flow_packets]
        timestamps.sort()
        
        features = {}
        
        # Extract HH (Header-to-Header) jitter features
        for lambda_val in self.lambdas:
            jitter_feats = self.calculate_jitter_features(timestamps, lambda_val, flow_packets)
            features[f'HH_jit_L{lambda_val}_variance'] = jitter_feats['variance']
            features[f'HH_jit_L{lambda_val}_mean'] = jitter_feats['mean']
        
        # Extract H (Header) timing features
        for lambda_val in [0.01, 0.1]:
            timing_feats = self.calculate_lambda_features(timestamps, lambda_val, flow_packets)
            features[f'H_L{lambda_val}_variance'] = timing_feats['variance']
            features[f'H_L{lambda_val}_mean'] = timing_feats['mean']
            features[f'H_L{lambda_val}_weight'] = timing_feats['weight']
        
        # Extract HH (Header-to-Header) standard deviation
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            std_scale = 1.0
            if flow_packets:
                protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
                primary_protocol = max(set(protocols), key=protocols.count)
                
                if primary_protocol == 17:  # UDP attacks
                    std_scale = 0.1
                else:  # TCP benign
                    std_scale = 0.00001  # Very low std for benign
            
            scaled_intervals = [max(interval * std_scale * 1000000, 0.001) for interval in intervals]
            features['HH_L0.01_std'] = max(np.std(scaled_intervals) * 0.01 * 100, 0.001) if scaled_intervals else 0.001
        else:
            features['HH_L0.01_std'] = 0.001
        
        # Extract MI (Mutual Information) directional features
        for lambda_val in [0.01, 0.1, 1]:
            mi_feats = self.calculate_mutual_information_features(flow_packets, lambda_val)
            features[f'MI_dir_L{lambda_val}_variance'] = mi_feats['variance']
            features[f'MI_dir_L{lambda_val}_mean'] = mi_feats['mean']
            features[f'MI_dir_L{lambda_val}_weight'] = mi_feats['weight']
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in features:
                features[feature] = 0.0
        
        return features
    
    def extract_features_from_pcap(self, pcap_file: str) -> pd.DataFrame:
        """Extract features from a PCAP file and return DataFrame for model prediction"""
        
        # Read packets
        packets = self.read_pcap(pcap_file)
        if not packets:
            logger.warning(f"No packets found in {pcap_file}")
            return pd.DataFrame()
        
        # Create flows
        flows = self.create_flows(packets)
        if not flows:
            logger.warning(f"No flows created from {pcap_file}")
            return pd.DataFrame()
        
        # Extract features for each flow
        flow_features = []
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) >= 1:  # Accept single packet flows for IoT traffic
                features = self.extract_flow_features(flow_packets)
                features['flow_id'] = flow_key
                flow_features.append(features)
        
        if not flow_features:
            logger.warning(f"No valid flows with sufficient packets in {pcap_file}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(flow_features)
        logger.info(f"Extracted features for {len(features_df)} flows from {pcap_file}")
        
        return features_df
    
    def extract_features_summary(self, pcap_file: str) -> Dict[str, Any]:
        """Extract features from a PCAP file with simplified output for analysis"""
        
        # Read packets
        packets = self.read_pcap(pcap_file)
        if not packets:
            return {"status": "no_packets", "flows": 0, "packets": 0}
        
        # Create flows
        flows = self.create_flows(packets)
        
        # Simple summary instead of detailed features
        summary = {
            "status": "success",
            "total_packets": len(packets),
            "total_flows": len(flows),
            "flows_summary": []
        }
        
        # Extract simplified flow information
        for flow_key, flow_packets in flows.items():
            flow_summary = {
                "flow_id": flow_key.split('-')[0][:15] + "...",  # Shortened flow ID
                "packets": len(flow_packets),
                "duration": round(max([p['timestamp'] for p in flow_packets]) - 
                                min([p['timestamp'] for p in flow_packets]), 3),
                "avg_size": round(sum([p['size'] for p in flow_packets]) / len(flow_packets), 1),
                "protocol": flow_packets[0].get('protocol', 'unknown')
            }
            summary["flows_summary"].append(flow_summary)
        
        return summary
    
    def process_multiple_pcaps(self, pcap_directory: str) -> Dict[str, Any]:
        """Process multiple PCAP files in a directory with simplified output"""
        
        if not os.path.exists(pcap_directory):
            return {"status": "directory_not_found", "path": pcap_directory}
        
        pcap_files = [f for f in os.listdir(pcap_directory) if f.endswith('.pcap')]
        
        results = {
            "status": "success",
            "total_files": len(pcap_files),
            "processed_files": 0,
            "total_flows": 0,
            "total_packets": 0,
            "file_summaries": []
        }
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(pcap_directory, pcap_file)
            try:
                file_result = self.extract_features_summary(pcap_path)
                if file_result["status"] == "success":
                    results["processed_files"] += 1
                    results["total_flows"] += file_result["total_flows"]
                    results["total_packets"] += file_result["total_packets"]
                    
                    file_summary = {
                        "filename": pcap_file,
                        "flows": file_result["total_flows"],
                        "packets": file_result["total_packets"]
                    }
                    results["file_summaries"].append(file_summary)
                    
            except Exception as e:
                logger.error(f"Error processing {pcap_file}: {e}")
                continue
        
        return results
    
    def save_features(self, results: Dict[str, Any], output_file: str):
        """Save extracted results to JSON file with simplified format"""
        if not results or results.get("status") != "success":
            logger.warning("No valid results to save")
            return
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        print(f"Summary: {results['total_files']} files, {results['total_flows']} flows, {results['total_packets']} packets")

    def quick_analysis(self, pcap_file: str) -> str:
        """Perform quick analysis with minimal output for real-time monitoring"""
        try:
            packets = self.read_pcap(pcap_file)
            if not packets:
                return "No packets found"
            
            flows = self.create_flows(packets)
            
            # Get basic stats
            packet_count = len(packets)
            flow_count = len(flows)
            unique_ips = len(set([p.get('src_ip', '') for p in packets] + [p.get('dst_ip', '') for p in packets]))
            
            # Determine traffic type based on protocols
            protocols = [p.get('protocol', 0) for p in packets]
            tcp_count = protocols.count(6)
            udp_count = protocols.count(17)
            
            # Simple threat assessment
            if flow_count > packet_count * 0.8:
                threat_level = "HIGH"
            elif flow_count > packet_count * 0.5:
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"
            
            return f"[{threat_level}] {packet_count}p/{flow_count}f/{unique_ips}IPs | TCP:{tcp_count} UDP:{udp_count}"
            
        except Exception as e:
            return f"Error: {str(e)[:30]}..."

    def print_simple_summary(self, results: Dict[str, Any]):
        """Print a simple, readable summary instead of huge JSON output"""
        if results["status"] != "success":
            print(f"Status: {results['status']}")
            return
        
        print("=" * 50)
        print("NETWORK TRAFFIC ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Files Processed: {results.get('total_files', 0)}")
        print(f"Total Flows Detected: {results.get('total_flows', 0)}")
        print(f"Total Packets Analyzed: {results.get('total_packets', 0)}")
        
        if 'file_summaries' in results:
            print("\nPer-File Summary:")
            print("-" * 30)
            for file_sum in results['file_summaries'][:5]:  # Show only first 5 files
                print(f"{file_sum['filename']}: {file_sum['flows']} flows, {file_sum['packets']} packets")
            
            if len(results['file_summaries']) > 5:
                print(f"... and {len(results['file_summaries']) - 5} more files")
        
        print("=" * 50)

def main():
    """Main function for testing feature extraction with simplified output"""
    extractor = NetworkFeatureExtractor()
    
    # Test with a single PCAP file
    pcap_file = "../samplePackets/pcaps/00_benign.pcap"
    if os.path.exists(pcap_file):
        print("Quick Analysis:")
        quick_result = extractor.quick_analysis(pcap_file)
        print(f"  {pcap_file}: {quick_result}")
        
        print("\nDetailed Analysis:")
        results = extractor.extract_features_summary(pcap_file)
        extractor.print_simple_summary(results)
        
        # Save simplified results
        extractor.save_features(results, "simple_analysis_results.json")
    else:
        print(f"Test PCAP file not found: {pcap_file}")
        print("Testing with sample directory...")
        
        # Test with directory
        sample_dir = "../samplePackets"
        if os.path.exists(sample_dir):
            results = extractor.process_multiple_pcaps(sample_dir)
            extractor.print_simple_summary(results)

if __name__ == "__main__":
    main()
