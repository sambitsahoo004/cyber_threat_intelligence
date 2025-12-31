#!/usr/bin/env python3
"""
Enhanced IOC Dataset Collection Script with Balanced Entity Extraction
Creates comprehensive training data with all entity types, using 'NULL' for missing entities
Addresses sparse and biased data issues for better model training
"""

import requests
import json
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import hashlib
from collections import defaultdict, Counter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BalancedIOCData:
    """Enhanced data class for balanced IOC training data"""
    text: str
    entities: Dict[str, List[str]]  # All entity types with values or 'NULL'
    source: str
    source_id: str
    created_date: str
    context: str
    confidence_scores: Dict[str, float]  # Confidence score for each entity type

class BalancedIOCPatterns:
    """Enhanced patterns with better coverage and context-aware extraction"""
    
    # Define all entity types we want to extract
    ALL_ENTITY_TYPES = [
        'IP', 'Domain', 'URL', 'File', 'Email', 'Type', 'Device', 'Vendor', 
        'Version', 'Software', 'Function', 'Platform', 'Malware', 'Vulnerability', 'ThreatActor', 'Other'
    ]
    
    # Enhanced patterns with better coverage
    ENHANCED_PATTERNS = {
        'IP': [
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6
            r'\b(?:IP\s+(?:address)?:?\s*)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
            r'\b(?:connects?\s+to|communicates?\s+with|beacons?\s+to)\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
        ],
        'Domain': [
            r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b',
            r'\b(?:domain|hostname|FQDN)(?:\s+is|\s*:)\s*([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})\b',
            r'\b(?:C2|command.{1,5}control|C&C)(?:\s+server)?\s+(?:at\s+)?([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})\b',
            r'\b(?:downloads?\s+from|connects?\s+to|queries|resolves)\s+([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})\b',
        ],
        'URL': [
            r'https?://[^\s<>"\']+',
            r'ftp://[^\s<>"\']+',
            r'\b(?:URL|link)(?:\s+is|\s*:)\s*(https?://[^\s<>"\']+)\b',
            r'\b(?:downloads?\s+from|fetches|requests)\s+(https?://[^\s<>"\']+)\b',
        ],
        'File': [
            r'\b[a-zA-Z0-9_\-\.]+\.(?:exe|dll|bat|cmd|scr|vbs|js|jar|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|sys|bin|so|dylib)\b',
            r'\b(?:file|filename|executable|binary)(?:\s+is|\s+name|\s*:)\s*([a-zA-Z0-9_\-\.]+\.(?:exe|dll|bat|cmd|scr|vbs|js|jar|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z))\b',
            r'\b(?:drops?|creates?|writes?|generates?)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_\-\.]+\.(?:exe|dll|bat|cmd|scr|vbs|js|jar|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z))\b',
            r'\b[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.(?:exe|dll|bat|cmd|scr|vbs|js|jar|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z)\b',
        ],
        'Email': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b(?:email|e-mail|sender|from)(?:\s+address|\s+is|\s*:)\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
        ],
        'Type': [
            # Attack/Malware types
            r'\b(?:malware|virus|worm|trojan|ransomware|spyware|adware|rootkit|bootkit|keylogger|stealer|infostealer|banker|cryptominer|miner|rat|backdoor|downloader|dropper|loader|botnet|zombie)\b',
            r'\b(?:phishing|spear.?phishing|whaling|social.engineering|watering.hole|supply.chain.attack|living.off.the.land|fileless|zero.day|apt|advanced.persistent.threat)\b',
            r'\b(?:DDoS|DoS|brute.?force|dictionary.attack|credential.stuffing|password.spraying|replay.attack|MITM|man.in.the.middle)\b',
            r'\b(?:this\s+is\s+a|identified\s+as\s+a|classified\s+as\s+a|type\s+of)\s+(malware|virus|trojan|ransomware|spyware|adware|rootkit|backdoor|phishing|attack)\b',
        ],
        'Device': [
            r'\b(?:router|switch|firewall|gateway|proxy|load.?balancer|server|workstation|laptop|desktop|endpoint|host|machine|computer|device|smartphone|tablet|mobile.device|IoT.device|embedded.system)\b',
            r'\b(?:web.server|database.server|mail.server|DNS.server|domain.controller|file.server|print.server|application.server)\b',
            r'\b(?:Windows.server|Linux.server|Unix.server|Apache.server|Nginx.server|IIS.server)\b',
            r'\b(?:targeting|affects|compromises|infects|targets)\s+(?:the\s+)?([a-zA-Z\s]+(?:server|device|system|machine|endpoint|router|switch|firewall)s?)\b',
        ],
        'Vendor': [
            r'\b(?:Microsoft|Apple|Google|Adobe|Oracle|IBM|Intel|AMD|Nvidia|Cisco|VMware|Amazon|Facebook|Meta|Twitter|LinkedIn|Salesforce|SAP|Symantec|McAfee|Trend.Micro|Kaspersky|Avast|AVG|Bitdefender|ESET|F-Secure|Sophos|CrowdStrike|FireEye|Mandiant|Palo.Alto|Fortinet|Check.Point|Juniper|HP|Dell|Lenovo|ASUS|Acer|Sony|Samsung|LG|Huawei|Xiaomi|OnePlus|Mozilla|Canonical|Red.Hat|SUSE|CentOS|Ubuntu|Debian|Android|iOS|macOS|Chrome|Firefox|Safari|Edge|Internet.Explorer)\b',
            r'\b(?:developed.by|created.by|made.by|from|by)\s+([A-Z][a-zA-Z\s&\.]{2,20}(?:Inc|Corp|Corporation|Ltd|Limited|Technologies|Systems|Software|Solutions|Security)?)\b',
        ],
        'Version': [
            r'\b(?:version|ver|v)\s*[\.\:]?\s*(\d+(?:\.\d+)*(?:[a-zA-Z]\d*)?)\b',
            r'\b([A-Za-z0-9\s]+)\s+(?:version\s+)?(\d+(?:\.\d+)*(?:\.[a-zA-Z]\d*)?)\b',
            r'\b(?:Windows\s+(?:XP|Vista|7|8|8\.1|10|11|Server\s+\d{4}(?:\s+R2)?)|macOS\s+\d+\.\d+|iOS\s+\d+\.\d+|Android\s+\d+\.\d+|Linux\s+kernel\s+\d+\.\d+)\b',
        ],
        'Software': [
            r'\b(?:Microsoft\s+Office|Adobe\s+(?:Reader|Acrobat|Flash|Photoshop|Illustrator)|Java|Oracle|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Apache|Nginx|IIS|Tomcat|WordPress|Drupal|Joomla|Magento|Shopify|Salesforce|SAP|AutoCAD|SolarWinds|TeamViewer|Zoom|Slack|Discord|Skype|Chrome|Firefox|Safari|Edge|Internet\s+Explorer|Visual\s+Studio|IntelliJ|Eclipse|Docker|Kubernetes|VirtualBox|VMware|Hyper-V)\b',
            r'\b(?:uses?|leverages?|exploits?|targets?|affects?)\s+([A-Z][a-zA-Z\s]+(?:software|application|program|tool|suite|platform|service))\b',
            r'\b(?:installed|running|executing|using)\s+([A-Z][a-zA-Z\s]+)\s+(?:software|application|program)\b',
        ],
        'Function': [
            r'\b(?:keylogging|screen.?capture|credential.theft|data.exfiltration|remote.access|backdoor.access|privilege.escalation|lateral.movement|persistence|evasion|defense.evasion|command.and.control|C2|C&C|data.encryption|file.encryption|system.monitoring|network.scanning|vulnerability.scanning|exploit|exploitation|payload.delivery|initial.access|reconnaissance|discovery|collection|impact|destruction|denial.of.service|cryptocurrency.mining|botnet.functionality)\b',
            r'\b(?:capable.of|performs|executes|carries.out|conducts|implements)\s+([a-zA-Z\s\-\.]+(?:logging|capture|theft|exfiltration|access|escalation|movement|persistence|evasion|scanning|mining|encryption))\b',
            r'\b(?:function|functionality|capability|feature|behavior|activity)(?:\s+is|\s+includes|\s*:)\s*([a-zA-Z\s\-\.]+)\b',
        ],
        'Platform': [
            r'\b(?:Windows|Linux|macOS|Mac\s+OS|Android|iOS|Unix|FreeBSD|OpenBSD|NetBSD|CentOS|Ubuntu|Red\s*Hat|RHEL|Debian|Fedora|SUSE|Arch|Mint|Kali|Parrot|BlackArch)\b',
            r'\b(?:AWS|Azure|Google\s+Cloud|GCP|Amazon\s+Web\s+Services|Microsoft\s+Azure|DigitalOcean|Linode|Vultr|Heroku|Cloudflare)\b',
            r'\b(?:x86|x64|x86_64|ARM|ARM64|MIPS|PowerPC|RISC-V)\b',
            r'\b(?:targets?|affects?|runs?.on|compatible.with|supports?)\s+([A-Za-z0-9\s]+(?:Windows|Linux|macOS|Android|iOS|Unix))\b',
        ],
        'Malware': [
            r'\b(?:Stuxnet|Conficker|WannaCry|NotPetya|Emotet|TrickBot|Qbot|IcedID|BazarLoader|Cobalt.Strike|Metasploit|Mimikatz|PowerShell.Empire|BloodHound|SharpHound|Rubeus|Kerberoast|ASREPRoast|Impacket|PsExec|WMIExec|SMBExec|DCSync|Golden.Ticket|Silver.Ticket|Skeleton.Key|DCShadow|NTDS\.dit|SAM|SYSTEM|SECURITY|Chrome\.exe|Firefox\.exe|svchost\.exe|explorer\.exe|winlogon\.exe|csrss\.exe|lsass\.exe|smss\.exe|wininit\.exe|spoolsv\.exe)\b',
            r'\b(?:known.as|identified.as|called|named|refers.to)\s+([A-Z][a-zA-Z0-9\-\.\_]+(?:bot|rat|stealer|miner|locker|crypter|packer|loader|dropper)?)\b',
            r'\b([A-Z][a-zA-Z0-9]{3,15})(?:\s+(?:malware|trojan|virus|ransomware|backdoor|rat))\b',
        ],
        'Vulnerability': [
            r'\bCVE-\d{4}-\d{4,7}\b',
            r'\b(?:buffer.overflow|stack.overflow|heap.overflow|integer.overflow|format.string|use.after.free|double.free|null.pointer.dereference|race.condition|injection|XSS|cross.site.scripting|CSRF|cross.site.request.forgery|SQL.injection|LDAP.injection|command.injection|code.injection|path.traversal|directory.traversal|file.inclusion|remote.file.inclusion|local.file.inclusion|XML.external.entity|XXE|server.side.request.forgery|SSRF|insecure.deserialization|broken.authentication|session.fixation|privilege.escalation|authentication.bypass|authorization.bypass|access.control|broken.access.control|security.misconfiguration|cryptographic.failure|insecure.cryptographic.storage|weak.cryptography|insufficient.logging|missing.security.headers)\b',
            r'\b(?:exploits?|targets?|leverages?)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z\s\-\.]+(?:vulnerability|flaw|weakness|bug|issue|exploit))\b',
            r'\b(?:zero.day|0.day|unpatched|known.vulnerability|security.flaw|critical.vulnerability|high.severity)\b',
        ],
        'ThreatActor': [
            # APT groups
            r'\b(?:APT|Advanced Persistent Threat)\s*\d+\b',
            r'\bAPT-?\d+\b',
            # Known threat actor naming patterns
            r'\b(?:Lazarus|Fancy Bear|Cozy Bear|APT28|APT29|Carbanak|FIN\d+|Turla|Sandworm|Equation Group|Comment Crew|Deep Panda|Axiom|Elderwood|Dragonfly|Energetic Bear|Sofacy|Pawn Storm|Sednit|Strontium|Wizard Spider|Grim Spider|TA\d+|UNC\d+|Winnti|Charming Kitten|OilRig|MuddyWater|Kimsuky|Konni|DarkHotel|Platinum|Stone Panda|Gothic Panda|Naikon|Lotus Blossom|BRONZE BUTLER|BlackTech|Machete|Volatile Cedar)\b',
            # Group/actor context patterns
            r'\b(?:threat actor|threat group|APT group|hacking group|cyber espionage group|nation-state actor|state-sponsored|hacker group|attack group)\s+(?:known as|called|named|identified as)\s+([A-Z][a-zA-Z0-9\s\-]+)\b',
            r'\b([A-Z][a-zA-Z\s]+)\s+(?:group|actor|team|crew)\s+(?:is responsible|has been|was identified|attributed)\b',
            # Attribution patterns
            r'\b(?:attributed to|linked to|associated with|believed to be)\s+(?:the\s+)?([A-Z][a-zA-Z0-9\s\-]+(?:\s+(?:group|actor|APT|team))?)\b',
        ],
        'Other': [
            # Hashes
            r'\b[a-fA-F0-9]{32}\b',  # MD5
            r'\b[a-fA-F0-9]{40}\b',  # SHA1
            r'\b[a-fA-F0-9]{64}\b',  # SHA256
            r'\b[a-fA-F0-9]{128}\b', # SHA512
            # Registry keys
            r'HKEY_[A-Z_]+\\[^\\]+(?:\\[^\\]+)*',
            # Process names
            r'\b[a-zA-Z0-9_\-\.]+\.exe\b',
            # Ports
            r'\b(?:port|ports?|listening.on|binds?.to)\s+(\d{1,5})\b',
            # Protocols
            r'\b(?:HTTP|HTTPS|FTP|SFTP|SSH|Telnet|SMTP|POP3|IMAP|DNS|DHCP|SNMP|SMB|CIFS|NFS|TCP|UDP|ICMP|SSL|TLS)\b',
            # File paths
            r'\b[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\b',
            r'\/(?:[^\/\s]+\/)*[^\/\s]*\b',  # Unix paths
        ]
    }
    
    # Context-aware patterns to improve extraction quality
    CONTEXT_PATTERNS = {
        'IP': [
            r'(?:connects?|communicates?|beacons?|sends?|transmits?|downloads?|uploads?|queries|resolves)\s+(?:to|from|with)\s+',
            r'(?:C2|command.{1,5}control|C&C)(?:\s+server)?\s+(?:at|on|is)\s+',
            r'(?:IP|address|host|server|endpoint)(?:\s+is|\s*:)\s*',
        ],
        'Domain': [
            r'(?:domain|hostname|FQDN|C2|command.{1,5}control)(?:\s+is|\s+server|\s*:)\s*',
            r'(?:connects?|queries|resolves|beacons?)\s+(?:to|with)\s+',
            r'(?:downloads?\s+from|fetches\s+from|requests\s+from)\s+',
        ],
        'File': [
            r'(?:drops?|creates?|writes?|generates?|saves?|stores?)\s+(?:the\s+)?(?:file\s+)?',
            r'(?:file|filename|executable|binary|payload)(?:\s+is|\s+name|\s*:)\s*',
            r'(?:executes?|runs?|launches?)\s+',
        ]
    }
    
    # Known false positives to filter out
    FALSE_POSITIVES = {
        'Domain': ['localhost', 'example.com', 'test.com', 'domain.com', 'sample.com', 'local', 'corp', 'internal'],
        'Email': ['user@example.com', 'test@test.com', 'admin@domain.com', 'support@company.com'],
        'IP': ['127.0.0.1', '0.0.0.0', '255.255.255.255', '192.168.1.1', '10.0.0.1'],
        'File': ['file.exe', 'sample.exe', 'test.dll', 'example.pdf', 'document.doc'],
        'Type': ['other', 'unknown', 'various', 'multiple', 'different', 'several'],
        'Device': ['system', 'service', 'application', 'generic', 'unknown'],
        'Platform': ['other', 'unknown', 'various', 'multiple', 'all', 'any'],
        'Software': ['software', 'application', 'program', 'tool', 'unknown'],
        'Vendor': ['vendor', 'company', 'organization', 'unknown', 'various'],
        'Version': ['version', 'unknown', 'latest', 'current', 'old', 'new'],
        'Function': ['function', 'functionality', 'feature', 'capability', 'unknown'],
        'Malware': ['malware', 'sample', 'unknown', 'generic', 'family'],
        'Vulnerability': ['vulnerability', 'flaw', 'weakness', 'issue', 'bug'],
        'ThreatActor': ['group', 'actor', 'unknown', 'threat', 'attacker', 'hacker', 'adversary', 'various', 'multiple', 'unidentified'],
        'Other': ['hash', 'value', 'data', 'information', 'content']
    }
    
    @classmethod
    def extract_balanced_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract entities ensuring all types are represented"""
        # Initialize all entity types with empty lists
        entities = {entity_type: [] for entity_type in cls.ALL_ENTITY_TYPES}
        
        # Extract entities for each type
        for entity_type in cls.ALL_ENTITY_TYPES:
            if entity_type in cls.ENHANCED_PATTERNS:
                patterns = cls.ENHANCED_PATTERNS[entity_type]
                extracted = set()  # Use set to avoid duplicates
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        entity_value = match.group().strip()
                        # Clean up the entity value
                        entity_value = cls._clean_entity_value(entity_value, entity_type)
                        if entity_value and cls._is_valid_entity(entity_value, entity_type):
                            extracted.add(entity_value)
                
                # Convert set to sorted list for consistency
                entities[entity_type] = sorted(list(extracted))
        
        # Apply context-aware improvements
        entities = cls._apply_context_improvements(text, entities)
        
        # Ensure no entity type is completely empty - fill with 'NULL' if needed
        for entity_type in entities:
            if not entities[entity_type]:
                entities[entity_type] = ['NULL']
            else:
                # Limit number of entities per type to avoid bias
                entities[entity_type] = entities[entity_type][:5]  # Max 5 per type
        
        return entities
    
    @classmethod
    def _clean_entity_value(cls, value: str, entity_type: str) -> str:
        """Clean and normalize entity values"""
        # Basic cleaning
        value = value.strip()
        value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
        
        # Type-specific cleaning
        if entity_type == 'Domain':
            # Remove protocol if present
            value = re.sub(r'^https?://', '', value)
            value = re.sub(r'^ftp://', '', value)
            # Remove trailing slash
            value = value.rstrip('/')
        elif entity_type == 'IP':
            # Remove any surrounding brackets
            value = value.strip('[](){}<>')
        elif entity_type == 'File':
            # Extract just the filename if it's a full path
            if '\\' in value:
                value = value.split('\\')[-1]
            elif '/' in value:
                value = value.split('/')[-1]
        elif entity_type in ['Type', 'Device', 'Platform', 'Software', 'Function']:
            # Normalize case
            value = value.lower()
            # Remove articles and prepositions
            value = re.sub(r'\b(?:the|a|an|and|or|of|in|on|at|to|for|with|by)\b\s*', '', value)
            value = value.strip()
        
        return value
    
    @classmethod
    def _is_valid_entity(cls, value: str, entity_type: str) -> bool:
        """Enhanced validation for extracted entities"""
        # Check false positives
        if entity_type in cls.FALSE_POSITIVES:
            if value.lower() in [fp.lower() for fp in cls.FALSE_POSITIVES[entity_type]]:
                return False
        
        # Basic validation rules
        if len(value.strip()) < 2:
            return False
            
        # Skip very generic terms
        if value.lower() in ['unknown', 'various', 'multiple', 'different', 'several', 'other', 'misc', 'general']:
            return False
        
        # Type-specific validation
        if entity_type == 'Domain':
            # Must have valid TLD and structure
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$', value):
                return False
            # Filter out localhost and private domains
            if value.lower() in ['localhost', 'local', 'localdomain', 'internal']:
                return False
        elif entity_type == 'IP':
            # Validate IP format
            parts = value.split('.')
            if len(parts) == 4 and all(part.isdigit() for part in parts):
                nums = [int(part) for part in parts]
                if all(0 <= num <= 255 for num in nums):
                    # Filter out localhost, broadcast, and other special IPs
                    if nums[0] in [0, 127, 255] or (nums[0] == 192 and nums[1] == 168):
                        return False
                    if value in ['0.0.0.0', '255.255.255.255', '127.0.0.1']:
                        return False
                    return True
            return False
        elif entity_type == 'Email':
            # Basic email validation
            if '@' not in value or value.count('@') != 1:
                return False
            local, domain = value.rsplit('@', 1)
            if not local or not domain or '.' not in domain:
                return False
        elif entity_type == 'File':
            # Must have valid file extension
            if '.' not in value:
                return False
            ext = value.split('.')[-1].lower()
            valid_exts = ['exe', 'dll', 'bat', 'cmd', 'scr', 'vbs', 'js', 'jar', 'pdf', 
                         'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'zip', 'rar', '7z', 
                         'sys', 'bin', 'so', 'dylib']
            if ext not in valid_exts:
                return False
        elif entity_type == 'Other':
            # For hashes, validate format
            if re.match(r'^[a-fA-F0-9]+$', value):
                if len(value) not in [32, 40, 64, 128]:  # MD5, SHA1, SHA256, SHA512
                    return False
        elif entity_type == 'ThreatActor':
            # Must be at least 3 characters
            if len(value) < 3:
                return False
            # Should start with capital letter or contain APT/FIN/TA/UNC
            if not (value[0].isupper() or any(prefix in value for prefix in ['APT', 'FIN', 'TA', 'UNC'])):
                return False
            # Filter out generic terms
            generic_terms = ['the group', 'threat actor', 'hacking group', 'unknown actor', 'adversary']
            if value.lower() in generic_terms:
                return False
        return True
    
    @classmethod
    def _apply_context_improvements(cls, text: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Apply context-aware improvements to entity extraction"""
        # Look for context clues to find missed entities
        
        # Look for IP addresses mentioned in context
        for pattern in cls.CONTEXT_PATTERNS.get('IP', []):
            context_matches = re.finditer(pattern + r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', text, re.IGNORECASE)
            for match in context_matches:
                ip = match.group(1)
                if cls._is_valid_entity(ip, 'IP') and ip not in entities['IP']:
                    entities['IP'].append(ip)
        
        # Look for domains mentioned in context
        for pattern in cls.CONTEXT_PATTERNS.get('Domain', []):
            context_matches = re.finditer(pattern + r'([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})', text, re.IGNORECASE)
            for match in context_matches:
                domain = match.group(1)
                if cls._is_valid_entity(domain, 'Domain') and domain not in entities['Domain']:
                    entities['Domain'].append(domain)
        
        # Look for files mentioned in context
        for pattern in cls.CONTEXT_PATTERNS.get('File', []):
            context_matches = re.finditer(pattern + r'([a-zA-Z0-9_\-\.]+\.(?:exe|dll|bat|cmd|scr|vbs|js|jar|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z))', text, re.IGNORECASE)
            for match in context_matches:
                filename = match.group(1)
                if cls._is_valid_entity(filename, 'File') and filename not in entities['File']:
                    entities['File'].append(filename)
        
        return entities
    
    @classmethod
    def calculate_confidence_scores(cls, text: str, entities: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate confidence scores for each entity type"""
        confidence_scores = {}
        
        for entity_type, entity_list in entities.items():
            if entity_list == ['NULL']:
                confidence_scores[entity_type] = 0.0
            else:
                # Base confidence on number of entities found and text length
                base_score = min(len(entity_list) / 3.0, 1.0)  # More entities = higher confidence
                
                # Adjust based on context keywords
                context_keywords = {
                    'IP': ['address', 'server', 'host', 'endpoint', 'connects', 'communicates'],
                    'Domain': ['domain', 'hostname', 'C2', 'command', 'control', 'queries'],
                    'File': ['file', 'executable', 'drops', 'creates', 'writes', 'payload'],
                    'Type': ['malware', 'attack', 'threat', 'campaign'],
                    'Device': ['device', 'system', 'machine', 'endpoint', 'server'],
                    'Platform': ['Windows', 'Linux', 'Android', 'iOS', 'platform'],
                    'Malware': ['malware', 'trojan', 'backdoor', 'ransomware', 'virus'],
                    'Vulnerability': ['CVE', 'vulnerability', 'exploit', 'flaw', 'weakness'],
                    'ThreatActor': ['APT', 'group', 'actor', 'attributed', 'campaign', 'threat', 'nation-state', 'espionage']
                }
                
                keyword_boost = 0.0
                if entity_type in context_keywords:
                    for keyword in context_keywords[entity_type]:
                        if keyword.lower() in text.lower():
                            keyword_boost += 0.1
                
                confidence_scores[entity_type] = min(base_score + keyword_boost, 1.0)
        
        return confidence_scores

class EnhancedOTXDataCollector:
    """Enhanced OTX collector with better data extraction"""
    
    def __init__(self, api_key: str, base_url: str = "https://otx.alienvault.com/api/v1/"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'X-OTX-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    def collect_pulses(self, limit: int = 1000, days_back: int = 120) -> List[Dict[str, Any]]:
        """Collect pulses from OTX with enhanced filtering"""
        logger.info(f"Collecting {limit} pulses from last {days_back} days")
        
        since = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%S')
        url = f"{self.base_url}pulses/subscribed"
        
        params = {
            'limit': limit,
            'modified_since': since
        }
        
        all_pulses = []
        page = 1
        
        try:
            while len(all_pulses) < limit:
                params['page'] = page
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                pulses = data.get('results', [])
                
                if not pulses:
                    break
                
                # Filter pulses for quality
                quality_pulses = self._filter_quality_pulses(pulses)
                all_pulses.extend(quality_pulses)
                
                logger.info(f"Page {page}: collected {len(quality_pulses)} quality pulses")
                page += 1
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting pulses: {e}")
        
        return all_pulses[:limit]
    
    def _filter_quality_pulses(self, pulses: List[Dict]) -> List[Dict]:
        """Filter pulses for quality and relevance"""
        quality_pulses = []
        
        for pulse in pulses:
            # Quality checks
            description = pulse.get('description', '')
            indicators = pulse.get('indicators', [])
            
            # Skip if too short or no indicators
            if len(description) < 50 or len(indicators) < 2:
                continue
                
            # Skip if mostly non-English
            if self._is_low_quality_text(description):
                continue
                
            # Check for diverse indicator types
            indicator_types = set(ind.get('type', '') for ind in indicators)
            if len(indicator_types) < 2:
                continue
                
            quality_pulses.append(pulse)
            
        return quality_pulses
    
    def _is_low_quality_text(self, text: str) -> bool:
        """Check if text is low quality"""
        # Check for minimum English content
        english_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text)
        
        if total_chars == 0:
            return True
            
        english_ratio = english_chars / total_chars
        return english_ratio < 0.7
    
    def extract_ioc_data(self, pulses: List[Dict]) -> List[BalancedIOCData]:
        """Extract balanced IOC data from pulses"""
        logger.info("Extracting balanced IOC data from pulses")
        
        ioc_data_list = []
        
        for pulse in pulses:
            try:
                # Combine description and references for richer context
                full_text = self._build_full_context(pulse)
                
                if len(full_text) < 100:  # Skip very short texts
                    continue
                
                # Extract entities using balanced approach
                entities = BalancedIOCPatterns.extract_balanced_entities(full_text)
                confidence_scores = BalancedIOCPatterns.calculate_confidence_scores(full_text, entities)
                
                # Create IOC data object
                ioc_data = BalancedIOCData(
                    text=full_text[:2000],  # Limit text length
                    entities=entities,
                    source="OTX",
                    source_id=pulse.get('id', ''),
                    created_date=pulse.get('created', ''),
                    context=pulse.get('description', '')[:500],
                    confidence_scores=confidence_scores
                )
                
                ioc_data_list.append(ioc_data)
                
            except Exception as e:
                logger.error(f"Error processing pulse {pulse.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Extracted {len(ioc_data_list)} IOC data entries")
        return ioc_data_list
    
    def _build_full_context(self, pulse: Dict) -> str:
        """Build full context from pulse data"""
        parts = []
        
        # Main description
        if pulse.get('description'):
            parts.append(pulse['description'])
        
        # Tags as context
        if pulse.get('tags'):
            tags_text = f"Tags: {', '.join(pulse['tags'])}"
            parts.append(tags_text)
        
        # Indicator information
        indicators = pulse.get('indicators', [])
        if indicators:
            ind_types = [ind.get('type', '') for ind in indicators[:10]]  # Limit to first 10
            ind_values = [ind.get('indicator', '') for ind in indicators[:10]]
            
            indicators_text = f"Indicators: {', '.join(f'{t}:{v}' for t, v in zip(ind_types, ind_values) if t and v)}"
            parts.append(indicators_text)
        
        # References
        if pulse.get('references'):
            refs = pulse['references'][:3]  # First 3 references
            refs_text = f"References: {', '.join(refs)}"
            parts.append(refs_text)
        
        return ' '.join(parts)

class MITREDataCollector:
    """Collector for MITRE ATT&CK data with IOC extraction"""
    
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/mitre/cti/master"
        self.session = requests.Session()
    
    def collect_attack_data(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Collect MITRE ATT&CK techniques and procedures"""
        logger.info(f"Collecting MITRE ATT&CK data (limit: {limit})")
        
        techniques = []
        
        try:
            # Get techniques from enterprise matrix
            techniques.extend(self._get_enterprise_techniques())
            
            # Get software/malware data
            techniques.extend(self._get_software_data())
            
            # Get groups data
            techniques.extend(self._get_groups_data())
            
        except Exception as e:
            logger.error(f"Error collecting MITRE data: {e}")
        
        return techniques[:limit]
    
    def _get_enterprise_techniques(self) -> List[Dict]:
        """Get enterprise techniques"""
        url = f"{self.base_url}/enterprise-attack/enterprise-attack.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            objects = data.get('objects', [])
            
            techniques = []
            for obj in objects:
                if obj.get('type') == 'attack-pattern':
                    technique_data = {
                        'id': obj.get('id', ''),
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'external_references': obj.get('external_references', []),
                        'x_mitre_platforms': obj.get('x_mitre_platforms', []),
                        'kill_chain_phases': obj.get('kill_chain_phases', []),
                        'type': 'technique'
                    }
                    techniques.append(technique_data)
            
            logger.info(f"Collected {len(techniques)} techniques")
            return techniques
            
        except Exception as e:
            logger.error(f"Error getting enterprise techniques: {e}")
            return []
    
    def _get_software_data(self) -> List[Dict]:
        """Get malware/software data"""
        url = f"{self.base_url}/enterprise-attack/enterprise-attack.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            objects = data.get('objects', [])
            
            software = []
            for obj in objects:
                if obj.get('type') in ['malware', 'tool']:
                    software_data = {
                        'id': obj.get('id', ''),
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'labels': obj.get('labels', []),
                        'x_mitre_platforms': obj.get('x_mitre_platforms', []),
                        'x_mitre_aliases': obj.get('x_mitre_aliases', []),
                        'type': 'software'
                    }
                    software.append(software_data)
            
            logger.info(f"Collected {len(software)} software entries")
            return software
            
        except Exception as e:
            logger.error(f"Error getting software data: {e}")
            return []
    
    def _get_groups_data(self) -> List[Dict]:
        """Get threat groups data"""
        url = f"{self.base_url}/enterprise-attack/enterprise-attack.json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            objects = data.get('objects', [])
            
            groups = []
            for obj in objects:
                if obj.get('type') == 'intrusion-set':
                    group_data = {
                        'id': obj.get('id', ''),
                        'name': obj.get('name', ''),
                        'description': obj.get('description', ''),
                        'aliases': obj.get('aliases', []),
                        'external_references': obj.get('external_references', []),
                        'type': 'group'
                    }
                    groups.append(group_data)
            
            logger.info(f"Collected {len(groups)} groups")
            return groups
            
        except Exception as e:
            logger.error(f"Error getting groups data: {e}")
            return []
    
    def extract_ioc_data(self, mitre_data: List[Dict]) -> List[BalancedIOCData]:
        """Extract IOC data from MITRE data"""
        logger.info("Extracting IOC data from MITRE data")
        
        ioc_data_list = []
        
        for item in mitre_data:
            try:
                # Build context from MITRE data
                full_text = self._build_mitre_context(item)
                
                if len(full_text) < 100:
                    continue
                
                # Extract entities
                entities = BalancedIOCPatterns.extract_balanced_entities(full_text)
                confidence_scores = BalancedIOCPatterns.calculate_confidence_scores(full_text, entities)
                
                # Create IOC data
                ioc_data = BalancedIOCData(
                    text=full_text[:2000],
                    entities=entities,
                    source="MITRE",
                    source_id=item.get('id', ''),
                    created_date=datetime.now().isoformat(),
                    context=item.get('description', '')[:500],
                    confidence_scores=confidence_scores
                )
                
                ioc_data_list.append(ioc_data)
                
            except Exception as e:
                logger.error(f"Error processing MITRE item {item.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Extracted {len(ioc_data_list)} IOC data entries from MITRE")
        return ioc_data_list
    
    def _build_mitre_context(self, item: Dict) -> str:
        """Build context from MITRE item"""
        parts = []
        
        # Name and description
        if item.get('name'):
            parts.append(f"Name: {item['name']}")
        
        if item.get('description'):
            parts.append(item['description'])
        
        # Platforms
        platforms = item.get('x_mitre_platforms', [])
        if platforms:
            parts.append(f"Platforms: {', '.join(platforms)}")
        
        # Aliases
        aliases = item.get('aliases', []) or item.get('x_mitre_aliases', [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
        
        # Labels (for software)
        labels = item.get('labels', [])
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        
        # Kill chain phases
        phases = item.get('kill_chain_phases', [])
        if phases:
            phase_names = [phase.get('phase_name', '') for phase in phases if phase.get('phase_name')]
            if phase_names:
                parts.append(f"Kill Chain: {', '.join(phase_names)}")
        
        return ' '.join(parts)
    
class DatasetBuilder:
    """Build and balance the final dataset"""
    
    def __init__(self):
        self.data_entries = []
    
    def add_data_entries(self, entries: List[BalancedIOCData]):
        """Add data entries to the dataset"""
        self.data_entries.extend(entries)
        logger.info(f"Added {len(entries)} entries. Total: {len(self.data_entries)}")
    
    def balance_dataset(self) -> List[BalancedIOCData]:
        """Balance the dataset to ensure good representation"""
        logger.info("Balancing dataset...")
        
        # Analyze entity distribution
        entity_stats = self._analyze_entity_distribution()
        logger.info(f"Entity distribution: {entity_stats}")
        
        # Balance by confidence scores
        balanced_entries = self._balance_by_confidence()
        
        # Ensure minimum representation for each entity type
        final_entries = self._ensure_minimum_representation(balanced_entries)
        
        logger.info(f"Final balanced dataset: {len(final_entries)} entries")
        return final_entries
    
    def _analyze_entity_distribution(self) -> Dict[str, int]:
        """Analyze distribution of entities across dataset"""
        stats = defaultdict(int)
        
        for entry in self.data_entries:
            for entity_type, entities in entry.entities.items():
                if entities != ['NULL']:
                    stats[entity_type] += len(entities)
        
        return dict(stats)
    
    def _balance_by_confidence(self) -> List[BalancedIOCData]:
        """Balance dataset by confidence scores"""
        # Sort by average confidence score
        scored_entries = []
        for entry in self.data_entries:
            avg_confidence = np.mean(list(entry.confidence_scores.values()))
            scored_entries.append((avg_confidence, entry))
        
        # Sort by confidence (high to low)
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        # Take top entries with good confidence scores
        threshold = 0.3  # Minimum average confidence
        good_entries = [entry for score, entry in scored_entries if score >= threshold]
        
        logger.info(f"Filtered to {len(good_entries)} entries with good confidence scores")
        return good_entries
    
    def _ensure_minimum_representation(self, entries: List[BalancedIOCData]) -> List[BalancedIOCData]:
        """Ensure minimum representation for each entity type"""
        # Count entries that have each entity type
        entity_counts = defaultdict(int)
        entity_entries = defaultdict(list)
        
        for entry in entries:
            for entity_type, entities in entry.entities.items():
                if entities != ['NULL']:
                    entity_counts[entity_type] += 1
                    entity_entries[entity_type].append(entry)
        
        # Find underrepresented entities
        min_count = max(50, len(entries) // 20)  # At least 5% representation
        
        # Use list instead of set and track IDs to avoid duplicates
        final_entries = list(entries)  # Start with all entries
        seen_ids = set(entry.source_id for entry in entries)
        
        for entity_type in BalancedIOCPatterns.ALL_ENTITY_TYPES:
            if entity_counts[entity_type] < min_count:
                logger.info(f"Entity type '{entity_type}' underrepresented: {entity_counts[entity_type]}/{min_count}")
                # Add more entries with this entity type from original data
                additional_needed = min_count - entity_counts[entity_type]
                additional_entries = self._find_additional_entries(entity_type, additional_needed, seen_ids)
                final_entries.extend(additional_entries)
                
                # Update seen_ids to track newly added entries
                for entry in additional_entries:
                    seen_ids.add(entry.source_id)
        
        return final_entries

    def _find_additional_entries(self, entity_type: str, count: int, seen_ids: set) -> List[BalancedIOCData]:
        """Find additional entries with specific entity type, avoiding duplicates"""
        candidates = []
        
        for entry in self.data_entries:
            # Skip if already included
            if entry.source_id in seen_ids:
                continue
                
            if entry.entities.get(entity_type, ['NULL']) != ['NULL']:
                candidates.append((entry.confidence_scores.get(entity_type, 0), entry))
        
        # Sort by confidence for this entity type
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [entry for score, entry in candidates[:count]]
    
    def save_dataset(self, filename: str = "balanced_ioc_dataset.json"):
        """Save the balanced dataset"""
        balanced_data = self.balance_dataset()
        
        # Convert to serializable format
        serializable_data = []
        for entry in balanced_data:
            data_dict = {
                'text': entry.text,
                'entities': entry.entities,
                'source': entry.source,
                'source_id': entry.source_id,
                'created_date': entry.created_date,
                'context': entry.context,
                'confidence_scores': entry.confidence_scores
            }
            serializable_data.append(data_dict)
        
        # Save to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filename}")
        
        # Generate statistics
        self._generate_dataset_stats(balanced_data, filename.replace('.json', '_stats.txt'))
        
        return filename
    
    def _generate_dataset_stats(self, data: List[BalancedIOCData], stats_filename: str):
        """Generate and save dataset statistics"""
        stats = []
        stats.append("=== BALANCED IOC DATASET STATISTICS ===\n")
        
        # Basic stats
        stats.append(f"Total entries: {len(data)}")
        stats.append(f"Sources: {set(entry.source for entry in data)}")
        stats.append("")
        
        # Entity distribution
        stats.append("=== ENTITY DISTRIBUTION ===")
        entity_counts = defaultdict(int)
        entity_with_values = defaultdict(int)
        
        for entry in data:
            for entity_type, entities in entry.entities.items():
                entity_counts[entity_type] += len(entities)
                if entities != ['NULL']:
                    entity_with_values[entity_type] += 1
        
        for entity_type in sorted(BalancedIOCPatterns.ALL_ENTITY_TYPES):
            total = entity_counts[entity_type]
            with_values = entity_with_values[entity_type]
            percentage = (with_values / len(data)) * 100
            stats.append(f"{entity_type:15}: {with_values:4} entries ({percentage:5.1f}%), {total:4} total values")
        
        stats.append("")
        
        # Confidence score distribution
        stats.append("=== CONFIDENCE SCORE DISTRIBUTION ===")
        all_scores = defaultdict(list)
        for entry in data:
            for entity_type, score in entry.confidence_scores.items():
                all_scores[entity_type].append(score)
        
        for entity_type in sorted(BalancedIOCPatterns.ALL_ENTITY_TYPES):
            scores = all_scores[entity_type]
            if scores:
                avg_score = np.mean(scores)
                min_score = min(scores)
                max_score = max(scores)
                stats.append(f"{entity_type:15}: avg={avg_score:.3f}, min={min_score:.3f}, max={max_score:.3f}")
        
        # Save stats
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stats))
        
        logger.info(f"Dataset statistics saved to {stats_filename}")

def main():
    """Main execution function"""
    # Configuration
    OTX_API_KEY = "50996bb794856bfda92b170aed5a68d21e3c9c76caf0bbe0544ac88c57ac7149"  # Replace with your actual API key
    OUTPUT_FILE = f"balanced_ioc_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Initialize dataset builder
    dataset_builder = DatasetBuilder()
    
    try:
        # Collect from OTX
        if OTX_API_KEY and OTX_API_KEY != "YOUR_OTX_API_KEY_HERE":
            logger.info("Starting OTX data collection...")
            otx_collector = EnhancedOTXDataCollector(OTX_API_KEY)
            otx_pulses = otx_collector.collect_pulses(limit=1000, days_back=500)
            otx_ioc_data = otx_collector.extract_ioc_data(otx_pulses)
            dataset_builder.add_data_entries(otx_ioc_data)
        else:
            logger.warning("OTX API key not provided, skipping OTX collection")
        
        # Collect from MITRE
        logger.info("Starting MITRE data collection...")
        mitre_collector = MITREDataCollector()
        mitre_data = mitre_collector.collect_attack_data(limit=500)
        mitre_ioc_data = mitre_collector.extract_ioc_data(mitre_data)
        dataset_builder.add_data_entries(mitre_ioc_data)
        
        # Save the balanced dataset
        output_file = dataset_builder.save_dataset(OUTPUT_FILE)
        logger.info(f"Dataset creation completed successfully!")
        logger.info(f"Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
