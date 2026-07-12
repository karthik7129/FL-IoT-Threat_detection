#!/usr/bin/env python3
"""
Enhanced IoT Network Traffic Simulator
Generates realistic network packets for specific attack types and benign traffic
using Scapy for federated learning training data.
"""

from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attack type mapping
ATTACK_NAMES = {
    0: "Benign",
    1: "Gafgyt Combo",
    2: "Gafgyt Junk",
    3: "Gafgyt TCP",
    5: "Mirai ACK",
    6: "Mirai Scan",
    7: "Mirai SYN",
    8: "Mirai UDP",
    9: "Mirai UDPPlain"
}

class EnhancedIoTTrafficSimulator:
    """
    Enhanced IoT network traffic simulator for generating specific attack patterns
    """
    
    def __init__(self):
        # Common IoT device IP ranges
        self.iot_subnets = [
            "192.168.1.0/24",
            "192.168.0.0/24", 
            "10.0.0.0/24",
            "172.16.0.0/24"
        ]
        
        # IoT device MAC address prefixes (real manufacturers)
        self.iot_mac_prefixes = [
            "00:1B:44",  # Netgear
            "B8:27:EB",  # Raspberry Pi
            "CC:32:E5",  # TP-Link
            "00:17:88",  # Philips Hue
            "18:B4:30",  # Nest
            "F0:EF:86",  # D-Link
            "00:90:A9",  # Western Digital
        ]
        
        self.device_profiles = {
            'smart_camera': {
                'ports': [80, 443, 554, 8080, 8181, 1935],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (64, 1500),
                'services': ['http', 'rtsp', 'onvif']
            },
            'smart_bulb': {
                'ports': [80, 443, 1883, 8883, 6667],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (40, 200),
                'services': ['http', 'mqtt', 'coap']
            },
            'router': {
                'ports': [22, 23, 53, 80, 443, 8080, 7547],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (40, 1500),
                'services': ['ssh', 'telnet', 'http', 'tr069']
            },
            'dvr': {
                'ports': [80, 8000, 8080, 37777, 34567],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (100, 1400),
                'services': ['http', 'proprietary']
            }
        }

    def generate_iot_ip(self, subnet: str = None) -> str:
        """Generate a random IoT device IP address"""
        if not subnet:
            subnet = random.choice(self.iot_subnets)
        
        base_ip = subnet.split('/')[0].rsplit('.', 1)[0]
        device_id = random.randint(10, 254)
        return f"{base_ip}.{device_id}"

    def generate_external_ip(self) -> str:
        """Generate external IP for internet traffic"""
        return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"

    def generate_mac_address(self) -> str:
        """Generate realistic IoT device MAC address"""
        prefix = random.choice(self.iot_mac_prefixes)
        suffix = ":".join([f"{random.randint(0, 255):02x}" for _ in range(3)])
        return f"{prefix}:{suffix}"

    def create_benign_traffic(self, num_packets: int = 500) -> List[Packet]:
        """Generate benign IoT traffic (Attack Type 0)"""
        packets = []
        logger.info(f"Generating {num_packets} benign packets")
        
        current_time = time.time()
        num_flows = num_packets // 10  # Create flows with ~10 packets each
        
        for flow_idx in range(num_flows):
            device_type = random.choice(list(self.device_profiles.keys()))
            profile = self.device_profiles[device_type]
            
            # Create flow 5-tuple
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip() if random.random() < 0.7 else self.generate_iot_ip()
            protocol = random.choice(profile['protocols'])
            src_port = random.randint(1024, 65535)
            dst_port = random.choice(profile['ports'])
            
            # Generate multiple packets for this flow
            packets_in_flow = random.randint(5, 15)
            for pkt_idx in range(packets_in_flow):
                if protocol == 'TCP':
                    flags = random.choice(["PA", "A", "S", "SA", "FA"])
                    packet = Ether(src=self.generate_mac_address()) / IP(src=src_ip, dst=dst_ip, ttl=random.randint(32, 128)) / TCP(
                        sport=src_port,
                        dport=dst_port,
                        flags=flags,
                        seq=random.randint(1, 4294967295),
                        ack=random.randint(1, 4294967295),
                        window=random.randint(1024, 65535)
                    )
                else:  # UDP
                    packet = Ether(src=self.generate_mac_address()) / IP(src=src_ip, dst=dst_ip, ttl=random.randint(32, 128)) / UDP(
                        sport=src_port,
                        dport=dst_port
                    )
                
                payload_size = random.randint(*profile['packet_sizes'])
                if payload_size > 60:
                    payload = self._generate_benign_payload(dst_port, payload_size - 60)
                    packet = packet / Raw(load=payload)
                
                # Benign traffic has variable inter-arrival times
                inter_arrival = random.uniform(0.001, 0.5)
                current_time += inter_arrival
                packet.time = current_time
                
                packets.append(packet)
                
                if len(packets) >= num_packets:
                    break
            
            if len(packets) >= num_packets:
                break
        
        return packets[:num_packets]

    def create_gafgyt_combo(self, num_packets: int = 500) -> List[Packet]:
        """Generate Gafgyt Combo attack (Attack Type 1)"""
        packets = []
        logger.info(f"Generating {num_packets} Gafgyt Combo attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_iot_ip() if random.random() < 0.6 else self.generate_external_ip()
            
            attack_variant = random.choice(['tcp_syn', 'udp_flood', 'http_flood'])
            
            if attack_variant == 'tcp_syn':
                packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(1, 64)) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([80, 8080, 443, 23, 22]),
                    flags="S",
                    seq=random.randint(1, 4294967295),
                    window=random.choice([1024, 2048, 4096, 8192])
                )
            elif attack_variant == 'udp_flood':
                packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(1, 64)) / UDP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([53, 80, 443, 123])
                ) / Raw(load=b"A" * random.randint(32, 512))
            else:  # http_flood
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=80,
                    flags="PA"
                ) / Raw(load=b"GET / HTTP/1.1\r\nHost: target\r\nUser-Agent: Gafgyt\r\n\r\n")
            
            inter_arrival = random.uniform(0.0001, 0.01)  # Fast attack
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_gafgyt_junk(self, num_packets: int = 500) -> List[Packet]:
        """Generate Gafgyt Junk attack (Attack Type 2)"""
        packets = []
        logger.info(f"Generating {num_packets} Gafgyt Junk attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # Junk data flooding
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(1, 32)) / UDP(
                sport=random.randint(1024, 65535),
                dport=random.randint(1, 65535)
            )
            
            # Random junk payload
            junk_size = random.randint(100, 1400)
            junk_data = bytes([random.randint(0, 255) for _ in range(junk_size)])
            packet = packet / Raw(load=junk_data)
            
            inter_arrival = random.uniform(0.00001, 0.001)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_gafgyt_tcp(self, num_packets: int = 500) -> List[Packet]:
        """Generate Gafgyt TCP attack (Attack Type 3)"""
        packets = []
        logger.info(f"Generating {num_packets} Gafgyt TCP attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_iot_ip() if random.random() < 0.5 else self.generate_external_ip()
            
            # TCP-specific Gafgyt attacks
            attack_type = random.choice(['syn_flood', 'ack_flood', 'push_flood'])
            
            if attack_type == 'syn_flood':
                flags = "S"
                seq = random.randint(1, 4294967295)
                ack = 0
            elif attack_type == 'ack_flood':
                flags = "A"
                seq = random.randint(1, 4294967295)
                ack = random.randint(1, 4294967295)
            else:  # push_flood
                flags = "PA"
                seq = random.randint(1, 4294967295)
                ack = random.randint(1, 4294967295)
            
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(1, 64)) / TCP(
                sport=random.randint(1024, 65535),
                dport=random.choice([80, 443, 8080, 23, 22, 25]),
                flags=flags,
                seq=seq,
                ack=ack,
                window=random.choice([0, 1024, 2048, 65535])
            )
            
            if attack_type == 'push_flood':
                packet = packet / Raw(load=b"X" * random.randint(10, 100))
            
            inter_arrival = random.uniform(0.0001, 0.005)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_mirai_ack(self, num_packets: int = 500) -> List[Packet]:
        """Generate Mirai ACK attack (Attack Type 5)"""
        packets = []
        logger.info(f"Generating {num_packets} Mirai ACK attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # Mirai ACK flooding
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(8, 64)) / TCP(
                sport=random.randint(1024, 65535),
                dport=random.choice([80, 443, 8080, 53, 22]),
                flags="A",
                seq=random.randint(1, 4294967295),
                ack=random.randint(1, 4294967295),
                window=random.choice([1024, 2048, 4096, 8192, 16384])
            )
            
            # Sometimes add small payload
            if random.random() < 0.3:
                packet = packet / Raw(load=b"ACKFLOOD" * random.randint(1, 8))
            
            inter_arrival = random.uniform(0.00001, 0.0001)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_mirai_scan(self, num_packets: int = 500) -> List[Packet]:
        """Generate Mirai Scan attack (Attack Type 6)"""
        packets = []
        logger.info(f"Generating {num_packets} Mirai Scan attack packets")
        
        scan_ports = [23, 2323, 80, 8080, 443, 8443, 7547, 5555, 9000]
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_iot_ip()  # Scanning local network
            
            # Rapid port scanning
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(32, 128)) / TCP(
                sport=random.randint(10000, 65535),
                dport=random.choice(scan_ports),
                flags="S",
                seq=random.randint(1, 4294967295),
                window=random.choice([1024, 2048, 4096, 5840, 8192])
            )
            
            # Add telnet brute force indicators for telnet ports
            if packet[TCP].dport in [23, 2323]:
                if random.random() < 0.4:
                    creds = random.choice([
                        b"admin\r\nadmin\r\n", b"root\r\nroot\r\n", 
                        b"user\r\nuser\r\n", b"admin\r\n123456\r\n"
                    ])
                    packet = packet / Raw(load=creds)
            
            inter_arrival = random.uniform(0.0001, 0.002)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_mirai_syn(self, num_packets: int = 500) -> List[Packet]:
        """Generate Mirai SYN attack (Attack Type 7)"""
        packets = []
        logger.info(f"Generating {num_packets} Mirai SYN attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # SYN flood attack
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(8, 64)) / TCP(
                sport=random.randint(1024, 65535),
                dport=random.choice([80, 443, 8080, 53, 22, 25, 110, 995]),
                flags="S",
                seq=random.randint(1, 4294967295),
                window=random.choice([1024, 2048, 4096, 5840, 8192, 16384])
            )
            
            # Add TCP options typical of Mirai
            if random.random() < 0.5:
                packet[TCP].options = [('MSS', random.choice([536, 1460, 1440]))]
            
            inter_arrival = random.uniform(0.00001, 0.0001)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_mirai_udp(self, num_packets: int = 500) -> List[Packet]:
        """Generate Mirai UDP attack (Attack Type 8)"""
        packets = []
        logger.info(f"Generating {num_packets} Mirai UDP attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # UDP flood with various payloads
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(8, 64)) / UDP(
                sport=random.randint(1024, 65535),
                dport=random.choice([53, 123, 1900, 5060, 80, 443])
            )
            
            # Different payload types
            payload_type = random.choice(['dns', 'ntp', 'ssdp', 'generic'])
            
            if payload_type == 'dns':
                # Fake DNS query
                payload = b'\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x01'
            elif payload_type == 'ntp':
                # NTP request
                payload = b'\x17\x00\x03\x2a' + b'\x00' * 44
            elif payload_type == 'ssdp':
                # SSDP M-SEARCH
                payload = b'M-SEARCH * HTTP/1.1\r\nHost: 239.255.255.250:1900\r\nMan: "ssdp:discover"\r\nST: upnp:rootdevice\r\nMX: 3\r\n\r\n'
            else:
                # Generic flood payload
                payload = b'MIRAI_UDP_FLOOD_' + bytes([random.randint(65, 90) for _ in range(random.randint(50, 500))])
            
            packet = packet / Raw(load=payload)
            
            inter_arrival = random.uniform(0.00001, 0.0001)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def create_mirai_udp_plain(self, num_packets: int = 500) -> List[Packet]:
        """Generate Mirai UDP Plain attack (Attack Type 9)"""
        packets = []
        logger.info(f"Generating {num_packets} Mirai UDP Plain attack packets")
        
        current_time = time.time()
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # Plain UDP flood with minimal payload
            packet = Ether() / IP(src=src_ip, dst=dst_ip, ttl=random.randint(8, 64)) / UDP(
                sport=random.randint(1024, 65535),
                dport=random.choice([80, 443, 53, 123, 1194, 4789])
            )
            
            # Simple repeating patterns
            patterns = [
                b'A' * random.randint(32, 128),
                b'0' * random.randint(32, 128),
                b'\x00' * random.randint(32, 128),
                b'\xFF' * random.randint(32, 128),
                b'FLOOD' * random.randint(10, 50)
            ]
            
            payload = random.choice(patterns)
            packet = packet / Raw(load=payload)
            
            inter_arrival = random.uniform(0.00001, 0.0001)
            current_time += inter_arrival
            packet.time = current_time
            
            packets.append(packet)
        
        return packets

    def _generate_benign_payload(self, port: int, size: int) -> bytes:
        """Generate realistic benign payload based on port"""
        if port == 80:
            # HTTP request/response
            payloads = [
                b"GET /api/status HTTP/1.1\r\nHost: device.local\r\nUser-Agent: IoTDevice/1.0\r\n\r\n",
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}",
                b"POST /config HTTP/1.1\r\nContent-Length: 50\r\n\r\n{\"setting\":\"value\"}"
            ]
            return random.choice(payloads)[:size]
        elif port == 443:
            # HTTPS (encrypted)
            return bytes([random.randint(0, 255) for _ in range(size)])
        elif port == 53:
            # DNS query/response
            return b'\x12\x34\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00'[:size]
        elif port == 1883:
            # MQTT
            return b'\x30\x12\x00\x04test{"temp":25.5}'[:size]
        else:
            # Generic application data
            return bytes([random.randint(32, 126) for _ in range(min(size, 100))])

    def save_packets_to_pcap(self, packets: List[Packet], filename: str):
        """Save generated packets to a PCAP file"""
        logger.info(f"Saving {len(packets)} packets to {filename}")
        wrpcap(filename, packets)

    def generate_all_attack_types(self, packets_per_type: int = 500):
        """Generate all attack types and save to separate files"""
        
        # Create output directory
        output_dir = "pcaps"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate each attack type
        attack_generators = {
            0: self.create_benign_traffic,
            1: self.create_gafgyt_combo,
            2: self.create_gafgyt_junk,
            3: self.create_gafgyt_tcp,
            5: self.create_mirai_ack,
            6: self.create_mirai_scan,
            7: self.create_mirai_syn,
            8: self.create_mirai_udp,
            9: self.create_mirai_udp_plain
        }
        
        all_packets = []
        
        for attack_id, generator_func in attack_generators.items():
            attack_name = ATTACK_NAMES[attack_id]
            print(f"Generating {packets_per_type} packets for {attack_name}...")
            
            packets = generator_func(packets_per_type)
            
            # Add attack label to each packet (as metadata)
            for packet in packets:
                packet.attack_label = attack_id
            
            # Save individual attack type
            filename = f"{output_dir}/{attack_id:02d}_{attack_name.lower().replace(' ', '_')}.pcap"
            self.save_packets_to_pcap(packets, filename)
            
            all_packets.extend(packets)
        
        # Shuffle and save combined dataset
        random.shuffle(all_packets)
        self.save_packets_to_pcap(all_packets, f"{output_dir}/combined_iot_dataset.pcap")
        
        return all_packets

def main():
    print("Enhanced IoT Network Traffic Simulator")
    print("=" * 50)
    
    simulator = EnhancedIoTTrafficSimulator()
    
    # Generate all attack types
    packets_per_type = 500
    print(f"Generating {packets_per_type} packets for each attack type...")
    print(f"Total packets to generate: {len(ATTACK_NAMES) * packets_per_type}")
    print()
    
    start_time = time.time()
    all_packets = simulator.generate_all_attack_types(packets_per_type)
    end_time = time.time()
    
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
    print(f"Total packets generated: {len(all_packets)}")
    print("\nFiles generated:")
    
    output_dir = "pcaps"
    for attack_id, attack_name in ATTACK_NAMES.items():
        filename = f"{attack_id:02d}_{attack_name.lower().replace(' ', '_')}.pcap"
        print(f"- {output_dir}/{filename} ({packets_per_type} packets)")
    
    print(f"- {output_dir}/combined_iot_dataset.pcap ({len(all_packets)} packets)")
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()