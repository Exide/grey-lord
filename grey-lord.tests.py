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
start_agent_client = grey_lord.start_agent_client
handle_train_command = grey_lord.handle_train_command
handle_analyze_command = grey_lord.handle_analyze_command
main = grey_lord.main


class TestGreyLordFunctions(unittest.TestCase):
    """Test the discrete functions in grey-lord.py"""
    
    def test_create_parser(self):
        """Test that the argument parser is created correctly."""
        parser = create_parser()
        
        # Test that it's an ArgumentParser
        self.assertIsNotNone(parser)
        
        # Test that it has the expected subcommands by checking if we can parse them
        # This tests the structure without triggering help output
        test_commands = ['agent', 'train', 'analyze', 'debug', 'optimize', 'data', 'config', 'model']
        
        # Test agent command
        args = parser.parse_args(['agent'])
        self.assertEqual(args.command, 'agent')
        
        # Test train command with basic args
        args = parser.parse_args(['train', '--epochs', '10'])
        self.assertEqual(args.command, 'train')
        self.assertEqual(args.epochs, 10)
    
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
    
    @patch('sys.path')
    def test_handle_train_command_missing_imports(self, mock_path):
        """Test the train command handler when training modules are missing."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.epochs = 10
        mock_args.batch_size = 4
        
        # The function should handle ImportError gracefully by calling sys.exit(1)
        with self.assertRaises(SystemExit) as cm:
            handle_train_command(mock_args)
        
        # Check that it exits with code 1 (error)
        self.assertEqual(cm.exception.code, 1)
    
    @patch('sys.path')
    def test_handle_analyze_command_missing_imports(self, mock_path):
        """Test the analyze command handler when analysis modules are missing."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.training_dir = 'test_dir'
        mock_args.output = 'test_output.png'
        mock_args.format = 'png'
        
        # The function should handle ImportError gracefully by calling sys.exit(1)
        with self.assertRaises(SystemExit) as cm:
            handle_analyze_command(mock_args)
        
        # Check that it exits with code 1 (error)
        self.assertEqual(cm.exception.code, 1)
    
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

    @patch.object(grey_lord, 'start_agent_client')
    @patch.object(grey_lord, 'create_parser')
    def test_main_agent_command(self, mock_create_parser, mock_start_agent):
        """Test main function routing to agent command."""
        # Mock parser and args
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.command = 'agent'
        mock_args.config = 'test_config.json'
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        mock_start_agent.return_value = 0
        
        with patch('sys.argv', ['grey-lord.py', 'agent']):
            result = main()
        
        mock_start_agent.assert_called_once_with('test_config.json')
        self.assertEqual(result, 0)

    @patch.object(grey_lord, 'create_parser')
    def test_main_no_args(self, mock_create_parser):
        """Test main function with no arguments shows help."""
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        with patch('sys.argv', ['grey-lord.py']):
            result = main()
        
        mock_parser.print_help.assert_called_once()
        self.assertEqual(result, 1)


if __name__ == '__main__':
    # Add the project root to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    unittest.main() 