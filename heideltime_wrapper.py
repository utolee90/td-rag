#!/usr/bin/env python3
"""
HeidelTime Python wrapper using subprocess
"""
import subprocess
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

class HeidelTimeWrapper:
    def __init__(self, heideltime_dir: str = "/mnt/nvme02/home/tdrag/vaiv/RTRAG/heideltime"):
        self.heideltime_dir = heideltime_dir
        self.jar_path = os.path.join(heideltime_dir, "target/de.unihd.dbs.heideltime.standalone.jar")
        self.lib_path = os.path.join(heideltime_dir, "lib/*")
        self.config_path = os.path.join(heideltime_dir, "config.props")
        
    def extract_temporal_expressions(self, text: str, language: str = "english", 
                                   document_type: str = "narratives", 
                                   pos_tagger: str = "no") -> Dict:
        """
        Extract temporal expressions from text using HeidelTime
        
        Args:
            text: Input text to process
            language: Language to use (default: english)
            document_type: Document type (narratives, news, colloquial, scientific)
            pos_tagger: POS tagger to use (stanfordpostagger, treetagger, hunpos, no)
            
        Returns:
            Dictionary containing original text, timeml output, and extracted timexes
        """
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_input:
            tmp_input.write(text)
            tmp_input_path = tmp_input.name
        
        try:
            # Build command
            cmd = [
                'java', '-cp', f"{self.jar_path}:{self.lib_path}",
                'de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone',
                tmp_input_path,
                '-c', self.config_path,
                '-l', language,
                '-t', document_type,
                '-pos', pos_tagger
            ]
            
            # Execute HeidelTime
            result = subprocess.run(
                cmd,
                cwd=self.heideltime_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"HeidelTime failed: {result.stderr}")
            
            # Parse XML output
            timeml_output = result.stdout
            timexes = self._parse_timexes(timeml_output)
            
            return {
                'original_text': text,
                'timeml_output': timeml_output,
                'timexes': timexes,
                'success': True
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("HeidelTime execution timed out")
        except Exception as e:
            return {
                'original_text': text,
                'timeml_output': None,
                'timexes': [],
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
    
    def _parse_timexes(self, timeml_output: str) -> List[Dict]:
        """Parse TIMEX3 tags from TimeML output"""
        timexes = []
        
        try:
            root = ET.fromstring(timeml_output)
            
            for timex in root.findall('.//TIMEX3'):
                timex_info = {
                    'tid': timex.get('tid'),
                    'type': timex.get('type'),
                    'value': timex.get('value'),
                    'text': timex.text or '',
                    'attributes': timex.attrib
                }
                timexes.append(timex_info)
                
        except ET.ParseError as e:
            print(f"Warning: Could not parse TimeML output: {e}")
            
        return timexes
    
    def process_file(self, input_file: str, output_file: str = None, **kwargs) -> Dict:
        """Process a text file with HeidelTime"""
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = self.extract_temporal_expressions(text, **kwargs)
        
        if output_file and result['success']:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['timeml_output'])
        
        return result


def main():
    # Example usage
    heideltime = HeidelTimeWrapper()
    
    # Test with sample text
    sample_text = "Today is July 13, 2025. I have a meeting tomorrow at 3 PM. Last week, I visited my grandmother."
    
    print("Processing text:", sample_text)
    print("-" * 50)
    
    result = heideltime.extract_temporal_expressions(sample_text)
    
    if result['success']:
        print("TimeML Output:")
        print(result['timeml_output'])
        print("\nExtracted Temporal Expressions:")
        for i, timex in enumerate(result['timexes'], 1):
            print(f"{i}. Text: '{timex['text']}' | Type: {timex['type']} | Value: {timex['value']}")
    else:
        print("Error:", result.get('error', 'Unknown error'))


if __name__ == "__main__":
    main()
