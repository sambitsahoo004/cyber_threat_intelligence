#!/usr/bin/env python3
"""
LLM-based Interpretation of CTI Predictions
Supports multiple free API providers:
- Google Gemini (Recommended - 1500 req/day free)
- Groq (Fast inference)
- Together AI
- OpenRouter
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LLMInterpreter")

# API client imports (install: pip install google-generativeai groq openai)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not available. Install: pip install google-generativeai")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    log.warning("groq not available. Install: pip install groq")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    log.warning("openai not available (needed for Together/OpenRouter). Install: pip install openai")

class LLMInterpreter:
    """Interpret CTI predictions using LLMs"""
    
    def __init__(self, 
                 predictions_file: str,
                 api_provider: str = 'gemini',
                 api_key: Optional[str] = None,
                 output_dir: str = 'llm_interpretations'):
        
        self.predictions_file = Path(predictions_file)
        self.api_provider = api_provider.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load predictions
        log.info(f"Loading predictions from {self.predictions_file}")
        with open(self.predictions_file, 'r') as f:
            self.predictions = json.load(f)
        
        # Initialize LLM client
        self.client = None
        self.model_name = None
        self._init_llm_client(api_key)
        
        log.info(f"✅ Initialized {self.api_provider} with model: {self.model_name}")
    
    def _init_llm_client(self, api_key: Optional[str]):
        """Initialize LLM client based on provider"""
        
        if self.api_provider == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ImportError("Install: pip install google-generativeai")
            
            if not api_key:
                raise ValueError("API key required. Get free key: https://makersuite.google.com/app/apikey")
            
            genai.configure(api_key=api_key)
            self.model_name = 'gemini-2.5-flash'  # Fast and free
            self.client = genai.GenerativeModel(self.model_name)
            
        elif self.api_provider == 'groq':
            if not GROQ_AVAILABLE:
                raise ImportError("Install: pip install groq")
            
            if not api_key:
                raise ValueError("API key required. Get free key: https://console.groq.com/keys")
            
            self.client = Groq(api_key=api_key)
            self.model_name = 'llama-3.1-70b-versatile'  # Fast and capable
            
        elif self.api_provider == 'together':
            if not OPENAI_AVAILABLE:
                raise ImportError("Install: pip install openai")
            
            if not api_key:
                raise ValueError("API key required. Get free key: https://api.together.xyz")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url='https://api.together.xyz/v1'
            )
            self.model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
            
        elif self.api_provider == 'openrouter':
            if not OPENAI_AVAILABLE:
                raise ImportError("Install: pip install openai")
            
            if not api_key:
                raise ValueError("API key required. Get free key: https://openrouter.ai/keys")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url='https://openrouter.ai/api/v1'
            )
            self.model_name = 'meta-llama/llama-3.1-8b-instruct:free'
            
        else:
            raise ValueError(f"Unknown provider: {self.api_provider}")
    
    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call LLM API (handles different providers)"""
        
        try:
            if self.api_provider == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text
            
            elif self.api_provider in ['groq', 'together', 'openrouter']:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity threat intelligence analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
        except Exception as e:
            log.error(f"LLM API call failed: {e}")
            return f"ERROR: {str(e)}"
    
    def find_node(self, node_id: Optional[int] = None, 
                  node_canonical: Optional[str] = None,
                  node_type: Optional[str] = None) -> Dict:
        """
        Find a node by ID, canonical name, or type
        Returns all predictions related to that node
        """
        results = {
            'node_info': None,
            'link_predictions': [],
            'node_classifications': [],
            'cluster_membership': []
        }
        
        # Search link predictions
        for edge_key, predictions in self.predictions.get('link_predictions', {}).items():
            for pred in predictions:
                subject = pred.get('subject', {})
                
                # Match by ID
                if node_id and subject.get('id') == node_id:
                    results['node_info'] = subject
                    results['link_predictions'].append(pred)
                
                # Match by canonical name (case-insensitive partial match)
                elif node_canonical and node_canonical.lower() in subject.get('canonical', '').lower():
                    results['node_info'] = subject
                    results['link_predictions'].append(pred)
                
                # Match by type
                elif node_type and subject.get('type') == node_type:
                    results['link_predictions'].append(pred)
        
        # Search node classifications
        for nt, classifications in self.predictions.get('node_classifications', {}).items():
            for pred in classifications:
                subject = pred.get('subject', {})
                
                if node_id and subject.get('id') == node_id:
                    results['node_info'] = subject
                    results['node_classifications'].append(pred)
                
                elif node_canonical and node_canonical.lower() in subject.get('canonical', '').lower():
                    results['node_info'] = subject
                    results['node_classifications'].append(pred)
                
                elif node_type and subject.get('type') == nt:
                    results['node_classifications'].append(pred)
        
        # Search cluster summaries
        for nt, clusters in self.predictions.get('cluster_summaries', {}).items():
            for cluster in clusters:
                for member in cluster.get('representative_members', []):
                    if node_id and member.get('node_id') == node_id:
                        results['node_info'] = member
                        results['cluster_membership'].append({
                            'cluster_id': cluster['cluster_id'],
                            'cluster_size': cluster['cluster_statistics']['size'],
                            'cohesion': cluster['cluster_statistics']['cohesion'],
                            'temporal_patterns': cluster.get('temporal_patterns', {}),
                            'common_relations': cluster.get('common_relations', [])
                        })
                    
                    elif node_canonical and node_canonical.lower() in member.get('canonical', '').lower():
                        results['node_info'] = member
                        results['cluster_membership'].append({
                            'cluster_id': cluster['cluster_id'],
                            'cluster_size': cluster['cluster_statistics']['size'],
                            'cohesion': cluster['cluster_statistics']['cohesion'],
                            'temporal_patterns': cluster.get('temporal_patterns', {}),
                            'common_relations': cluster.get('common_relations', [])
                        })
        
        return results
    
    def build_prompt(self, node_data: Dict) -> str:
        """Build LLM prompt from node predictions"""
        
        node_info = node_data['node_info']
        if not node_info:
            return "No node information found."
        
        prompt = f"""# Cyber Threat Intelligence Analysis

## Target Node
- **Type**: {node_info.get('type', 'Unknown')}
- **Identifier**: {node_info.get('canonical', 'Unknown')}
- **Node ID**: {node_info.get('id', 'Unknown')}
- **First Seen**: {node_info.get('first_seen', 'Unknown')}
- **Last Seen**: {node_info.get('last_seen', 'Unknown')}
- **Frequency**: {node_info.get('frequency', 0)} occurrences

---

## 1. Link Predictions (Potential New Connections)
"""
        
        # Link predictions
        if node_data['link_predictions']:
            prompt += f"\nFound {len(node_data['link_predictions'])} predicted relationships:\n\n"
            
            for i, pred in enumerate(node_data['link_predictions'][:10], 1):  # Top 10
                obj = pred.get('object', {})
                prompt += f"**{i}. {pred.get('predicted_relation', 'related_to')} → {obj.get('canonical', 'Unknown')}**\n"
                prompt += f"   - Confidence: {pred.get('confidence', 0):.3f}\n"
                prompt += f"   - Object Type: {obj.get('type', 'Unknown')}\n"
                
                # Evidence
                evidence = pred.get('evidence', [])
                if evidence:
                    prompt += f"   - Evidence:\n"
                    for ev in evidence[:3]:
                        prompt += f"      * {ev.get('type', 'unknown')}: {ev.get('explanation', 'N/A')}\n"
                
                # Meta-path support
                mp_support = pred.get('meta_path_support', [])
                if mp_support:
                    prompt += f"   - Meta-path support: {mp_support[0].get('path_pattern', 'N/A')}\n"
                
                prompt += "\n"
        else:
            prompt += "\n*No link predictions available.*\n\n"
        
        # Node classification
        prompt += "\n## 2. Node Classification\n"
        
        if node_data['node_classifications']:
            for pred in node_data['node_classifications']:
                prompt += f"- **Predicted Class**: {pred.get('predicted_class', 'Unknown')}\n"
                prompt += f"- **Confidence**: {pred.get('confidence', 0):.3f}\n"
                
                if 'true_class' in pred:
                    prompt += f"- **True Class**: {pred.get('true_class', 'Unknown')}\n"
                    prompt += f"- **Correct**: {pred.get('correct', False)}\n"
                
                nearest = pred.get('nearest_labeled_node', {})
                if nearest:
                    prompt += f"- **Similar to**: {nearest.get('canonical', 'Unknown')} (same class)\n"
                
                prompt += "\n"
        else:
            prompt += "\n*No classification predictions available.*\n\n"
        
        # Cluster membership
        prompt += "\n## 3. Behavioral Clustering\n"
        
        if node_data['cluster_membership']:
            for cluster in node_data['cluster_membership']:
                prompt += f"**Cluster {cluster['cluster_id']}**\n"
                prompt += f"- Size: {cluster['cluster_size']} similar nodes\n"
                prompt += f"- Cohesion: {cluster['cohesion']:.3f}\n"
                
                # Temporal patterns
                temporal = cluster.get('temporal_patterns', {})
                if temporal:
                    prompt += f"- Activity period: {temporal.get('earliest_activity', 'Unknown')} to {temporal.get('latest_activity', 'Unknown')}\n"
                
                # Common relations
                relations = cluster.get('common_relations', [])
                if relations:
                    prompt += f"- Common relations:\n"
                    for rel in relations[:5]:
                        prompt += f"   * {rel.get('relation', 'Unknown')}: {rel.get('count', 0)} times\n"
                
                prompt += "\n"
        else:
            prompt += "\n*No cluster information available.*\n\n"
        
        # Analysis request
        prompt += """
---

## Task: CTI Analysis

As a cybersecurity threat intelligence analyst, provide:

1. **Threat Assessment** (1-2 sentences)
   - Overall threat level and nature of this IOC

2. **Key Findings** (bullet points)
   - Most significant predicted relationships
   - Behavioral patterns from clustering
   - Classification insights

3. **Actionable Recommendations** (numbered list)
   - Specific actions for security teams
   - Investigation priorities
   - Mitigation strategies

4. **Confidence Assessment**
   - Reliability of these predictions
   - Areas needing further investigation

Keep your response concise, technical, and actionable.
"""
        
        return prompt
    
    def interpret_node(self, 
                      node_id: Optional[int] = None,
                      node_canonical: Optional[str] = None,
                      node_type: Optional[str] = None,
                      save_output: bool = True) -> Dict:
        """
        Generate LLM interpretation for a single node
        
        Args:
            node_id: Node ID
            node_canonical: Canonical name (partial match OK)
            node_type: Node type filter
            save_output: Save interpretation to file
        
        Returns:
            Dict with interpretation results
        """
        
        log.info(f"🔍 Searching for node: id={node_id}, canonical={node_canonical}, type={node_type}")
        
        # Find node and gather predictions
        node_data = self.find_node(node_id, node_canonical, node_type)
        
        if not node_data['node_info']:
            log.error("❌ Node not found in predictions")
            return {'error': 'Node not found'}
        
        log.info(f"✅ Found node: {node_data['node_info'].get('canonical', 'Unknown')}")
        log.info(f"   - Link predictions: {len(node_data['link_predictions'])}")
        log.info(f"   - Classifications: {len(node_data['node_classifications'])}")
        log.info(f"   - Clusters: {len(node_data['cluster_membership'])}")
        
        # Build prompt
        prompt = self.build_prompt(node_data)
        
        log.info(f"📤 Sending to {self.api_provider} LLM...")
        
        # Call LLM
        interpretation = self._call_llm(prompt)
        
        # Prepare results
        results = {
            'node_info': node_data['node_info'],
            'query_time': datetime.now().isoformat(),
            'api_provider': self.api_provider,
            'model': self.model_name,
            'raw_data': {
                'link_predictions_count': len(node_data['link_predictions']),
                'classifications_count': len(node_data['node_classifications']),
                'cluster_memberships': len(node_data['cluster_membership'])
            },
            'llm_interpretation': interpretation,
            'full_prompt': prompt
        }
        
        # Save to file
        if save_output:
            node_identifier = node_data['node_info'].get('canonical', 'unknown').replace('/', '_')
            output_file = self.output_dir / f"interpretation_{node_identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            log.info(f"💾 Saved interpretation to: {output_file}")
            
            # Also save markdown version
            md_file = output_file.with_suffix('.md')
            with open(md_file, 'w') as f:
                f.write(f"# CTI Analysis Report\n\n")
                f.write(f"**Node**: {node_data['node_info'].get('canonical', 'Unknown')}\n\n")
                f.write(f"**Generated**: {results['query_time']}\n\n")
                f.write(f"**Model**: {self.model_name}\n\n")
                f.write("---\n\n")
                f.write(interpretation)
            
            log.info(f"📝 Saved markdown to: {md_file}")
        
        return results
    
    def interpret_multiple(self, 
                          node_list: List[Dict],
                          batch_delay: float = 1.0) -> List[Dict]:
        """
        Interpret multiple nodes (with rate limiting)
        
        Args:
            node_list: List of dicts with 'node_id', 'node_canonical', or 'node_type'
            batch_delay: Delay between API calls (seconds)
        
        Returns:
            List of interpretation results
        """
        import time
        
        results = []
        
        for i, node_query in enumerate(node_list, 1):
            log.info(f"\n{'='*60}")
            log.info(f"Processing {i}/{len(node_list)}")
            
            result = self.interpret_node(**node_query)
            results.append(result)
            
            # Rate limiting
            if i < len(node_list):
                time.sleep(batch_delay)
        
        return results
    
    def list_available_nodes(self, node_type: Optional[str] = None, limit: int = 50):
        """List available nodes from predictions"""
        
        nodes = set()
        
        # From link predictions
        for edge_key, predictions in self.predictions.get('link_predictions', {}).items():
            for pred in predictions[:limit]:
                subject = pred.get('subject', {})
                if not node_type or subject.get('type') == node_type:
                    nodes.add((
                        subject.get('id'),
                        subject.get('canonical'),
                        subject.get('type')
                    ))
        
        # From classifications
        for nt, classifications in self.predictions.get('node_classifications', {}).items():
            if node_type and nt != node_type:
                continue
            
            for pred in classifications[:limit]:
                subject = pred.get('subject', {})
                nodes.add((
                    subject.get('id'),
                    subject.get('canonical'),
                    subject.get('type')
                ))
        
        print(f"\n📋 Available nodes (showing up to {limit}):")
        print(f"{'ID':<10} {'Type':<15} {'Canonical'}")
        print("-" * 70)
        
        for node_id, canonical, ntype in sorted(nodes)[:limit]:
            print(f"{str(node_id):<10} {str(ntype):<15} {canonical}")
        
        return list(nodes)


def main():
    parser = argparse.ArgumentParser(
        description='Interpret CTI predictions using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Interpret by canonical name (partial match):
   python interpret_with_llm.py \\
     --predictions predictions/predictions_for_llm.json \\
     --api-provider gemini \\
     --api-key YOUR_KEY \\
     --canonical "malicious.com"

2. Interpret by node ID:
   python interpret_with_llm.py \\
     --predictions predictions/predictions_for_llm.json \\
     --api-provider groq \\
     --api-key YOUR_KEY \\
     --node-id 12345

3. List available nodes:
   python interpret_with_llm.py \\
     --predictions predictions/predictions_for_llm.json \\
     --list-nodes \\
     --node-type Domain

Free API Keys:
- Gemini: https://makersuite.google.com/app/apikey (1500 req/day)
- Groq: https://console.groq.com/keys (fast inference)
- Together AI: https://api.together.xyz
- OpenRouter: https://openrouter.ai/keys
        """
    )
    
    parser.add_argument('--predictions', required=True, 
                       help='Path to predictions_for_llm.json')
    
    parser.add_argument('--api-provider', 
                       choices=['gemini', 'groq', 'together', 'openrouter'],
                       default='gemini',
                       help='LLM API provider (default: gemini)')
    
    parser.add_argument('--api-key', 
                       help='API key (or set GEMINI_API_KEY / GROQ_API_KEY env var)')
    
    parser.add_argument('--node-id', type=int,
                       help='Node ID to interpret')
    
    parser.add_argument('--canonical',
                       help='Node canonical name (partial match OK)')
    
    parser.add_argument('--node-type',
                       help='Node type filter')
    
    parser.add_argument('--output', default='llm_interpretations',
                       help='Output directory (default: llm_interpretations)')
    
    parser.add_argument('--list-nodes', action='store_true',
                       help='List available nodes and exit')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output to file')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    
    # Initialize interpreter
    interpreter = LLMInterpreter(
        predictions_file=args.predictions,
        api_provider=args.api_provider,
        api_key=api_key,
        output_dir=args.output
    )
    
    # List nodes and exit
    if args.list_nodes:
        interpreter.list_available_nodes(node_type=args.node_type)
        return
    
    # Check if query provided
    if not args.node_id and not args.canonical and not args.node_type:
        print("❌ Error: Provide --node-id, --canonical, or --node-type")
        print("   Or use --list-nodes to see available nodes")
        return
    
    # Interpret node
    result = interpreter.interpret_node(
        node_id=args.node_id,
        node_canonical=args.canonical,
        node_type=args.node_type,
        save_output=not args.no_save
    )
    
    if 'error' in result:
        print(f"\n❌ {result['error']}")
        print("\nTry --list-nodes to see available nodes")
        return
    
    # Print interpretation
    print("\n" + "="*70)
    print("🤖 LLM INTERPRETATION")
    print("="*70)
    print(f"\n{result['llm_interpretation']}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()