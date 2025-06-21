#!/usr/bin/env python3
"""
Tests for Grey Lord main functions

Simple tests to verify the discrete functions work correctly.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Import the functions we want to test
import importlib.util
spec = importlib.util.spec_from_file_location("grey_lord", "grey-lord.py")
grey_lord = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grey_lord)

# Import the functions we want to test
setup_logging = grey_lord.setup_logging
create_parser = grey_lord.create_parser
route_to_training_cli = grey_lord.route_to_training_cli
route_to_tools_cli = grey_lord.route_to_tools_cli
start_agent_client = grey_lord.start_agent_client


class TestGreyLordFunctions(unittest.TestCase):
    """Test the discrete functions in grey-lord.py"""
    
    def test_create_parser(self):
        """Test that the argument parser is created correctly."""
        parser = create_parser()
        
        # Test that it's an ArgumentParser
        self.assertIsNotNone(parser)
        
        # Test that it has the expected subcommands
        # Parse help to check subcommands exist
        with patch('sys.stderr'):  # Suppress help output
            try:
                parser.parse_args(['--help'])
            except SystemExit:
                pass  # Expected for --help
    
    def test_setup_logging(self):
        """Test that logging is set up correctly."""
        with patch('logging.basicConfig') as mock_config:
            setup_logging()
            mock_config.assert_called_once()
            
            # Check that it was called with expected parameters
            call_args = mock_config.call_args
            self.assertIn('level', call_args.kwargs)
            self.assertIn('format', call_args.kwargs)
            self.assertIn('handlers', call_args.kwargs)
    
    @patch('subprocess.run')
    def test_route_to_training_cli(self, mock_run):
        """Test routing to training CLI."""
        mock_run.return_value.returncode = 0
        
        result = route_to_training_cli(['--epochs', '10'])
        
        # Check that subprocess.run was called
        mock_run.assert_called_once()
        
        # Check the command structure
        call_args = mock_run.call_args[0][0]
        self.assertTrue(call_args[1].endswith('train.py'))
        self.assertIn('--epochs', call_args)
        self.assertIn('10', call_args)
        
        # Check return code
        self.assertEqual(result, 0)
    
    @patch('subprocess.run')
    def test_route_to_tools_cli(self, mock_run):
        """Test routing to tools CLI."""
        mock_run.return_value.returncode = 0
        
        result = route_to_tools_cli(['analyze', '--help'])
        
        # Check that subprocess.run was called
        mock_run.assert_called_once()
        
        # Check the command structure
        call_args = mock_run.call_args[0][0]
        self.assertTrue(call_args[1].endswith('tools.py'))
        self.assertIn('analyze', call_args)
        self.assertIn('--help', call_args)
        
        # Check return code
        self.assertEqual(result, 0)
    
    @patch('agent.telnet_client.TelnetClient')
    def test_start_agent_client_success(self, mock_client_class):
        """Test starting the agent client successfully."""
        # Mock the client instance
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        result = start_agent_client('test_config.json')
        
        # Check that client was created and started
        mock_client_class.assert_called_once_with('test_config.json')
        mock_client.start.assert_called_once()
        
        # Check return code
        self.assertEqual(result, 0)
    
    @patch('agent.telnet_client.TelnetClient')
    def test_start_agent_client_keyboard_interrupt(self, mock_client_class):
        """Test handling KeyboardInterrupt in agent client."""
        # Mock the client to raise KeyboardInterrupt
        mock_client = MagicMock()
        mock_client.start.side_effect = KeyboardInterrupt()
        mock_client_class.return_value = mock_client
        
        result = start_agent_client('test_config.json')
        
        # Should return 0 for graceful exit
        self.assertEqual(result, 0)
    
    @patch('agent.telnet_client.TelnetClient')
    def test_start_agent_client_exception(self, mock_client_class):
        """Test handling exceptions in agent client."""
        # Mock the client to raise an exception
        mock_client_class.side_effect = Exception("Test error")
        
        result = start_agent_client('test_config.json')
        
        # Should return 1 for error
        self.assertEqual(result, 1)


if __name__ == '__main__':
    # Add the project root to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    unittest.main() 