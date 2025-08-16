#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# File: digi-infrastructure/port-validator.py
# Purpose: Validate port allocations and detect conflicts across all services
# Related: port-mapping.yml, docker-compose files
# Tags: port-validation, conflict-detection, infrastructure
# -----------------------------------------------------------------------------

import yaml
import socket
import subprocess
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortAllocation:
    """Represents a port allocation for a service."""
    external: int
    internal: int
    service: str
    description: str
    status: str = "active"
    conflict_with: Optional[str] = None

class PortValidator:
    """
    Validates port allocations and detects conflicts across all services.
    
    This tool ensures that:
    1. No external ports conflict with each other
    2. No external ports conflict with system ports
    3. All required ports are available
    4. Port ranges are properly allocated
    """
    
    def __init__(self, port_mapping_file: str = "port-mapping.yml"):
        self.port_mapping_file = Path(port_mapping_file)
        self.port_allocations: Dict[str, PortAllocation] = {}
        self.conflicts: List[Dict[str, Any]] = []
        self.system_ports: Dict[int, str] = {}
        
    def load_port_mapping(self) -> bool:
        """Load port mapping configuration from YAML file."""
        try:
            with open(self.port_mapping_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract all port allocations
            for category, services in config.items():
                if isinstance(services, dict):
                    for service_name, service_config in services.items():
                        if isinstance(service_config, dict) and 'external' in service_config:
                            allocation = PortAllocation(
                                external=service_config['external'],
                                internal=service_config['internal'],
                                service=service_config['service'],
                                description=service_config.get('description', ''),
                                status=service_config.get('status', 'active'),
                                conflict_with=service_config.get('conflict_with')
                            )
                            self.port_allocations[f"{category}.{service_name}"] = allocation
            
            logger.info(f"Loaded {len(self.port_allocations)} port allocations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load port mapping: {e}")
            return False
    
    def detect_port_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between external ports."""
        conflicts = []
        external_ports = {}
        
        for service_id, allocation in self.port_allocations.items():
            if allocation.status == 'active':
                if allocation.external in external_ports:
                    conflict = {
                        'type': 'external_port_conflict',
                        'port': allocation.external,
                        'service1': external_ports[allocation.external],
                        'service2': service_id,
                        'severity': 'high'
                    }
                    conflicts.append(conflict)
                    logger.error(f"Port conflict: {service_id} and {external_ports[allocation.external]} both use port {allocation.external}")
                else:
                    external_ports[allocation.external] = service_id
        
        self.conflicts = conflicts
        return conflicts
    
    def check_system_port_conflicts(self) -> List[Dict[str, Any]]:
        """Check if any ports conflict with system-reserved ports."""
        system_conflicts = []
        
        # Common system ports to avoid
        system_ports = {
            22: "SSH",
            23: "Telnet", 
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP Alternative",
            8443: "HTTPS Alternative"
        }
        
        for service_id, allocation in self.port_allocations.items():
            if allocation.status == 'active':
                if allocation.external in system_ports:
                    conflict = {
                        'type': 'system_port_conflict',
                        'port': allocation.external,
                        'service': service_id,
                        'system_service': system_ports[allocation.external],
                        'severity': 'medium'
                    }
                    system_conflicts.append(conflict)
                    logger.warning(f"System port conflict: {service_id} uses port {allocation.external} (reserved for {system_ports[allocation.external]})")
        
        self.conflicts.extend(system_conflicts)
        return system_conflicts
    
    def check_port_availability(self) -> List[Dict[str, Any]]:
        """Check if ports are actually available on the system."""
        availability_issues = []
        
        for service_id, allocation in self.port_allocations.items():
            if allocation.status == 'active':
                if not self.is_port_available(allocation.external):
                    issue = {
                        'type': 'port_unavailable',
                        'port': allocation.external,
                        'service': service_id,
                        'severity': 'high'
                    }
                    availability_issues.append(issue)
                    logger.error(f"Port {allocation.external} is not available for {service_id}")
        
        self.conflicts.extend(availability_issues)
        return availability_issues
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available on the system."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return True  # Assume available if we can't check
    
    def check_docker_port_conflicts(self) -> List[Dict[str, Any]]:
        """Check for port conflicts in running Docker containers."""
        docker_conflicts = []
        
        try:
            # Get running containers and their port mappings
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}:{{.Ports}}'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        container_name, ports = line.split(':', 1)
                        if ports and ports != '':
                            # Parse port mappings like "0.0.0.0:8000->8000/tcp"
                            for port_mapping in ports.split(', '):
                                if '->' in port_mapping:
                                    external_port = port_mapping.split('->')[0].split(':')[1]
                                    try:
                                        external_port = int(external_port)
                                        
                                        # Check if this conflicts with our planned allocations
                                        for service_id, allocation in self.port_allocations.items():
                                            if (allocation.status == 'active' and 
                                                allocation.external == external_port and
                                                allocation.service != container_name):
                                                conflict = {
                                                    'type': 'docker_port_conflict',
                                                    'port': external_port,
                                                    'planned_service': service_id,
                                                    'running_container': container_name,
                                                    'severity': 'high'
                                                }
                                                docker_conflicts.append(conflict)
                                                logger.error(f"Docker port conflict: {container_name} uses port {external_port} planned for {service_id}")
                                    except (ValueError, IndexError):
                                        continue
            
        except Exception as e:
            logger.warning(f"Could not check Docker port conflicts: {e}")
        
        self.conflicts.extend(docker_conflicts)
        return docker_conflicts
    
    def validate_port_ranges(self) -> List[Dict[str, Any]]:
        """Validate that ports are allocated within appropriate ranges."""
        range_violations = []
        
        for service_id, allocation in self.port_allocations.items():
            if allocation.status == 'active':
                port = allocation.external
                
                # Check if port is within expected ranges
                if port < 1024 and port not in [5432, 6379]:  # Allow common DB ports
                    violation = {
                        'type': 'privileged_port_violation',
                        'port': port,
                        'service': service_id,
                        'severity': 'medium',
                        'recommendation': f'Move {service_id} to port >= 1024'
                    }
                    range_violations.append(violation)
                    logger.warning(f"Privileged port violation: {service_id} uses port {port}")
                
                # Check if port is within our defined ranges
                if 1000 <= port <= 1999 and not service_id.startswith('infrastructure'):
                    violation = {
                        'type': 'range_violation',
                        'port': port,
                        'service': service_id,
                        'expected_range': '2000-2999',
                        'severity': 'low'
                    }
                    range_violations.append(violation)
                    logger.info(f"Range violation: {service_id} uses port {port} outside expected range")
        
        self.conflicts.extend(range_violations)
        return range_violations
    
    def generate_resolution_plan(self) -> Dict[str, Any]:
        """Generate a plan to resolve all detected conflicts."""
        resolution_plan = {
            'summary': {
                'total_conflicts': len(self.conflicts),
                'high_severity': len([c for c in self.conflicts if c.get('severity') == 'high']),
                'medium_severity': len([c for c in self.conflicts if c.get('severity') == 'medium']),
                'low_severity': len([c for c in self.conflicts if c.get('severity') == 'low'])
            },
            'resolutions': []
        }
        
        # Group conflicts by type
        conflicts_by_type = {}
        for conflict in self.conflicts:
            conflict_type = conflict['type']
            if conflict_type not in conflicts_by_type:
                conflicts_by_type[conflict_type] = []
            conflicts_by_type[conflict_type].append(conflict)
        
        # Generate resolutions for each conflict type
        for conflict_type, conflicts in conflicts_by_type.items():
            if conflict_type == 'external_port_conflict':
                resolution_plan['resolutions'].append({
                    'type': 'port_reallocation',
                    'description': f'Reallocate conflicting ports for {len(conflicts)} services',
                    'actions': [
                        'Identify services that can be moved to different ports',
                        'Update port-mapping.yml with new allocations',
                        'Update docker-compose files with new port mappings',
                        'Restart affected services'
                    ]
                })
            
            elif conflict_type == 'docker_port_conflict':
                resolution_plan['resolutions'].append({
                    'type': 'container_management',
                    'description': f'Resolve conflicts with {len(conflicts)} running containers',
                    'actions': [
                        'Stop conflicting containers',
                        'Update container port mappings',
                        'Restart containers with new configurations'
                    ]
                })
            
            elif conflict_type == 'system_port_conflict':
                resolution_plan['resolutions'].append({
                    'type': 'port_reallocation',
                    'description': f'Move {len(conflicts)} services away from system ports',
                    'actions': [
                        'Identify alternative ports in appropriate ranges',
                        'Update service configurations',
                        'Test new port allocations'
                    ]
                })
        
        return resolution_plan
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete port validation."""
        logger.info("Starting port validation...")
        
        if not self.load_port_mapping():
            return {'error': 'Failed to load port mapping'}
        
        # Run all validation checks
        self.detect_port_conflicts()
        self.check_system_port_conflicts()
        self.check_port_availability()
        self.check_docker_port_conflicts()
        self.validate_port_ranges()
        
        # Generate resolution plan
        resolution_plan = self.generate_resolution_plan()
        
        validation_result = {
            'status': 'failed' if self.conflicts else 'passed',
            'conflicts': self.conflicts,
            'resolution_plan': resolution_plan,
            'port_allocations': {
                service_id: {
                    'external': alloc.external,
                    'internal': alloc.internal,
                    'service': alloc.service,
                    'description': alloc.description,
                    'status': alloc.status
                }
                for service_id, alloc in self.port_allocations.items()
            }
        }
        
        logger.info(f"Port validation completed. Status: {validation_result['status']}")
        if self.conflicts:
            logger.error(f"Found {len(self.conflicts)} conflicts")
        else:
            logger.info("No port conflicts detected")
        
        return validation_result

def main():
    """Main function to run port validation."""
    validator = PortValidator()
    result = validator.run_full_validation()
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Exit with error code if conflicts found
    if result.get('status') == 'failed':
        exit(1)

if __name__ == "__main__":
    main()
