import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, CircularProgress, Paper } from '@mui/material';
import Editor from '@monaco-editor/react';

const CodeEditor = ({ 
  initialCode = '', 
  language = 'python', 
  theme = 'vs-dark',
  height = '400px',
  readOnly = false,
  onCodeChange = () => {},
  onRunCode = () => {}
}) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isEditorReady, setIsEditorReady] = useState(false);

  useEffect(() => {
    setCode(initialCode);
  }, [initialCode]);

  const handleEditorDidMount = () => {
    setIsEditorReady(true);
  };

  const handleEditorChange = (value) => {
    setCode(value);
    onCodeChange(value);
  };

  const handleRunCode = async () => {
    setIsRunning(true);
    setOutput('Running code...');
    
    try {
      // In a real implementation, this would send the code to a backend service
      // For now, we'll simulate execution with a timeout
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock execution result based on language
      let result;
      if (language === 'python') {
        if (code.includes('print(')) {
          // Extract what's inside print statements
          const printMatches = code.match(/print\((.*?)\)/g);
          if (printMatches) {
            result = printMatches
              .map(match => {
                const content = match.substring(6, match.length - 1);
                // Handle string literals
                if (content.startsWith('"') || content.startsWith("'")) {
                  return content.substring(1, content.length - 1);
                }
                // Simple evaluation for numbers
                try {
                  return eval(content);
                } catch {
                  return content;
                }
              })
              .join('\n');
          } else {
            result = "Code executed successfully (no output)";
          }
        } else {
          result = "Code executed successfully (no output)";
        }
      } else if (language === 'javascript') {
        try {
          // Very simple and unsafe evaluation - in a real app, this would use a sandboxed environment
          const consoleOutput = [];
          const originalConsoleLog = console.log;
          console.log = (...args) => {
            consoleOutput.push(args.join(' '));
          };
          
          eval(code);
          
          console.log = originalConsoleLog;
          result = consoleOutput.join('\n') || "Code executed successfully (no output)";
        } catch (error) {
          result = `Error: ${error.message}`;
        }
      } else {
        result = "Code execution for this language is not supported in the preview.";
      }
      
      setOutput(result);
      onRunCode(result);
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Box sx={{ mb: 4 }}>
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Interactive Code Editor
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {readOnly 
            ? 'This is a read-only code example.' 
            : 'Edit the code below and run it to see the output.'}
        </Typography>
      </Box>
      
      <Box sx={{ border: 1, borderColor: 'grey.300', borderRadius: 1, overflow: 'hidden' }}>
        {!isEditorReady && (
          <Box sx={{ height: height, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <CircularProgress size={40} />
          </Box>
        )}
        
        <Editor
          height={height}
          language={language}
          value={code}
          theme={theme}
          onChange={handleEditorChange}
          onMount={handleEditorDidMount}
          options={{
            readOnly,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 14,
            fontFamily: 'Fira Code, monospace',
            automaticLayout: true,
          }}
        />
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', p: 1, bgcolor: 'grey.100' }}>
          <Typography variant="caption" sx={{ alignSelf: 'center' }}>
            Language: {language.charAt(0).toUpperCase() + language.slice(1)}
          </Typography>
          
          {!readOnly && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleRunCode}
              disabled={isRunning || !isEditorReady}
              startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {isRunning ? 'Running...' : 'Run Code'}
            </Button>
          )}
        </Box>
      </Box>
      
      {(output || !readOnly) && (
        <Paper 
          variant="outlined" 
          sx={{ 
            mt: 2, 
            p: 2, 
            bgcolor: 'grey.900', 
            color: 'grey.100',
            fontFamily: 'Fira Code, monospace',
            fontSize: '0.875rem',
            minHeight: '100px',
            maxHeight: '200px',
            overflow: 'auto'
          }}
        >
          <Typography variant="caption" display="block" sx={{ mb: 1, color: 'grey.500' }}>
            Output:
          </Typography>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
            {output || 'Run the code to see output here.'}
          </pre>
        </Paper>
      )}
    </Box>
  );
};

export default CodeEditor;
