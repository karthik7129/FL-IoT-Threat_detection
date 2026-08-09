#!/usr/bin/env python3
"""
Test script for IoT Network Traffic Simulator
Demonstrates basic functionality and creates sample datasets
"""

import os
import sys
import logging
from samplePackets.packet_simulator import IoTTrafficSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simulator():
    """Test the IoT traffic simulator with various scenarios"""
    
    logger.info("Initializing IoT Traffic Simulator")
    simulator = IoTTrafficSimulator()
    
    # Create output directory
    output_dir = "generated_traffic"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Test 1: Generate normal traffic for different device types
    logger.info("Test 1: Generating normal traffic for different IoT devices")
    
    device_types = ['smart_camera', 'smart_bulb', 'router', 'smart_thermostat', 'baby_monitor']
    
    for device_type in device_types:
        logger.info(f"Generating normal traffic for {device_type}")
        packets = simulator.create_normal_traffic(num_packets=200, device_type=device_type)
        filename = os.path.join(output_dir, f"normal_{device_type}.pcap")
        simulator.save_packets_to_pcap(packets, filename)
        logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Test 2: Generate attack traffic
    logger.info("Test 2: Generating attack traffic patterns")
    
    # Mirai attack phases
    mirai_phases = ['scan', 'infection', 'ddos', 'mixed']
    for phase in mirai_phases:
        logger.info(f"Generating Mirai {phase} traffic")
        packets = simulator.create_mirai_attack(num_packets=300, attack_phase=phase)
        filename = os.path.join(output_dir, f"mirai_{phase}.pcap")
        simulator.save_packets_to_pcap(packets, filename)
        logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Gafgyt attack
    logger.info("Generating Gafgyt attack traffic")
    packets = simulator.create_gafgyt_attack(num_packets=300)
    filename = os.path.join(output_dir, "gafgyt_attack.pcap")
    simulator.save_packets_to_pcap(packets, filename)
    logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Tsunami attack
    logger.info("Generating Tsunami attack traffic")
    packets = simulator.create_tsunami_attack(num_packets=300)
    filename = os.path.join(output_dir, "tsunami_attack.pcap")
    simulator.save_packets_to_pcap(packets, filename)
    logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Test 3: Generate mixed datasets for training
    logger.info("Test 3: Generating mixed datasets for ML training")
    
    # Small training set
    logger.info("Generating small training dataset")
    packets = simulator.generate_mixed_dataset(total_packets=1000, attack_ratio=0.2)
    filename = os.path.join(output_dir, "training_small.pcap")
    simulator.save_packets_to_pcap(packets, filename)
    logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Medium training set
    logger.info("Generating medium training dataset")
    packets = simulator.generate_mixed_dataset(total_packets=5000, attack_ratio=0.3)
    filename = os.path.join(output_dir, "training_medium.pcap")
    simulator.save_packets_to_pcap(packets, filename)
    logger.info(f"Saved {len(packets)} packets to {filename}")
    
    # Test 4: Generate balanced datasets for each client in federated learning
    logger.info("Test 4: Generating datasets for federated learning clients")
    
    num_clients = 3
    for client_id in range(1, num_clients + 1):
        logger.info(f"Generating dataset for FL Client {client_id}")
        
        # Each client gets different attack ratios to simulate real-world scenarios
        attack_ratios = [0.1, 0.3, 0.5]
        attack_ratio = attack_ratios[client_id - 1]
        
        packets = simulator.generate_mixed_dataset(total_packets=2000, attack_ratio=attack_ratio)
        filename = os.path.join(output_dir, f"fl_client_{client_id}_data.pcap")
        simulator.save_packets_to_pcap(packets, filename)
        logger.info(f"Saved {len(packets)} packets for Client {client_id} (attack ratio: {attack_ratio})")
    
    logger.info("All tests completed successfully!")
    logger.info(f"Generated traffic files are saved in '{output_dir}' directory")
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total files generated: {len(os.listdir(output_dir))}")
    print("\nFile types generated:")
    print("- Normal traffic for 5 IoT device types")
    print("- Mirai attack patterns (4 phases)")
    print("- Gafgyt attack patterns")
    print("- Tsunami attack patterns")
    print("- Mixed training datasets (2 sizes)")
    print("- Federated learning client datasets (3 clients)")
    print("\nNext steps:")
    print("1. Install Wireshark to view the PCAP files")
    print("2. Extract features from PCAP files")
    print("3. Use the datasets for federated learning training")
    print("="*60)

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import scapy
        logger.info("Scapy is available")
        return True
    except ImportError:
        logger.error("Scapy is not installed. Please run: pip install -r requirements_simulator.txt")
        return False

def main():
    """Main function to run the test"""
    print("IoT Network Traffic Simulator - Test Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        test_simulator()
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
