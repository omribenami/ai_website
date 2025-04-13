import React, { useState, useEffect } from 'react';
import { Box, TextField, InputAdornment, IconButton, Typography, Paper, List, ListItem, ListItemText, Divider, Chip } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';

const SearchComponent = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchHistory, setSearchHistory] = useState([]);

  useEffect(() => {
    // Load search history from localStorage in a real implementation
    const mockHistory = ['neural networks', 'python setup', 'tensorflow vs pytorch'];
    setSearchHistory(mockHistory);
  }, []);

  const handleSearch = async () => {
    if (!searchTerm.trim()) return;
    
    setIsSearching(true);
    
    try {
      // In a real implementation, this would call an API endpoint
      // For now, we'll simulate a search with mock data
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Mock search results based on the search term
      const mockResults = generateMockResults(searchTerm);
      
      setSearchResults(mockResults);
      
      // Add to search history if not already present
      if (!searchHistory.includes(searchTerm)) {
        const newHistory = [searchTerm, ...searchHistory].slice(0, 5);
        setSearchHistory(newHistory);
        // In a real implementation, save to localStorage or backend
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const generateMockResults = (term) => {
    const lowerTerm = term.toLowerCase();
    
    // Base set of potential results
    const allResults = [
      {
        id: 'result-1',
        title: 'Introduction to Neural Networks',
        type: 'module',
        path: '/learn/deep-learning/neural-networks',
        excerpt: 'Learn about the fundamental building blocks of neural networks, including neurons, layers, and activation functions.',
        matches: ['neural', 'networks']
      },
      {
        id: 'result-2',
        title: 'Setting Up Your Python Environment',
        type: 'module',
        path: '/learn/environment-setup/python',
        excerpt: 'Step-by-step guide to setting up Python for AI development, including virtual environments and package management.',
        matches: ['python', 'setup', 'environment']
      },
      {
        id: 'result-3',
        title: 'TensorFlow vs. PyTorch: Detailed Comparison',
        type: 'section',
        path: '/learn/deep-learning/frameworks/comparison',
        excerpt: 'An in-depth comparison of the two most popular deep learning frameworks: TensorFlow and PyTorch.',
        matches: ['tensorflow', 'pytorch', 'comparison']
      },
      {
        id: 'result-4',
        title: 'Convolutional Neural Networks for Image Classification',
        type: 'module',
        path: '/learn/deep-learning/cnn',
        excerpt: 'Understand how CNNs work and how to implement them for image classification tasks.',
        matches: ['neural', 'networks', 'cnn', 'convolutional']
      },
      {
        id: 'result-5',
        title: 'Recurrent Neural Networks and LSTM',
        type: 'module',
        path: '/learn/deep-learning/rnn',
        excerpt: 'Learn about RNNs, LSTMs, and their applications in sequence modeling tasks.',
        matches: ['neural', 'networks', 'rnn', 'recurrent']
      },
      {
        id: 'result-6',
        title: 'Python Libraries for AI Development',
        type: 'section',
        path: '/learn/python-for-ai/libraries',
        excerpt: 'Overview of essential Python libraries for AI development, including NumPy, pandas, and scikit-learn.',
        matches: ['python', 'libraries']
      },
      {
        id: 'result-7',
        title: 'Setting Up GPU Acceleration with CUDA',
        type: 'section',
        path: '/learn/environment-setup/gpu',
        excerpt: 'Configure your RTX 3080 GPU for deep learning with CUDA and cuDNN.',
        matches: ['gpu', 'setup', 'cuda', 'rtx']
      },
      {
        id: 'result-8',
        title: 'TensorFlow Installation and Configuration',
        type: 'section',
        path: '/learn/deep-learning/frameworks/tensorflow-setup',
        excerpt: 'Step-by-step guide to installing and configuring TensorFlow with GPU support.',
        matches: ['tensorflow', 'setup', 'installation']
      },
      {
        id: 'result-9',
        title: 'PyTorch Installation and Configuration',
        type: 'section',
        path: '/learn/deep-learning/frameworks/pytorch-setup',
        excerpt: 'Step-by-step guide to installing and configuring PyTorch with GPU support.',
        matches: ['pytorch', 'setup', 'installation']
      }
    ];
    
    // Filter results based on search term
    return allResults.filter(result => {
      // Check if search term matches title, excerpt, or any of the match keywords
      return (
        result.title.toLowerCase().includes(lowerTerm) ||
        result.excerpt.toLowerCase().includes(lowerTerm) ||
        result.matches.some(match => match.includes(lowerTerm) || lowerTerm.includes(match))
      );
    });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const clearSearch = () => {
    setSearchTerm('');
    setSearchResults([]);
  };

  const handleHistoryItemClick = (term) => {
    setSearchTerm(term);
    // Trigger search with the selected history item
    setTimeout(() => handleSearch(), 0);
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 800, mx: 'auto', my: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Search Course Content
        </Typography>
        
        <Box sx={{ display: 'flex', mb: 3 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search for topics, modules, or keywords..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyPress={handleKeyPress}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
              endAdornment: searchTerm && (
                <InputAdornment position="end">
                  <IconButton
                    aria-label="clear search"
                    onClick={clearSearch}
                    edge="end"
                  >
                    <ClearIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Box>
        
        {searchHistory.length > 0 && !searchResults.length && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Recent Searches
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {searchHistory.map((term, index) => (
                <Chip
                  key={index}
                  label={term}
                  onClick={() => handleHistoryItemClick(term)}
                  clickable
                  size="small"
                />
              ))}
            </Box>
          </Box>
        )}
        
        {isSearching ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography>Searching...</Typography>
          </Box>
        ) : searchResults.length > 0 ? (
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              {searchResults.length} results found for "{searchTerm}"
            </Typography>
            
            <List>
              {searchResults.map((result) => (
                <React.Fragment key={result.id}>
                  <ListItem 
                    alignItems="flex-start" 
                    button 
                    component="a" 
                    href={result.path}
                    sx={{ 
                      borderRadius: 1,
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="subtitle1" component="div">
                            {result.title}
                          </Typography>
                          <Chip 
                            label={result.type === 'module' ? 'Module' : 'Section'} 
                            size="small"
                            color={result.type === 'module' ? 'primary' : 'secondary'}
                            sx={{ ml: 1 }}
                          />
                        </Box>
                      }
                      secondary={
                        <React.Fragment>
                          <Typography
                            component="span"
                            variant="body2"
                            color="text.primary"
                            sx={{ display: 'block', mt: 1 }}
                          >
                            {result.excerpt}
                          </Typography>
                          <Typography
                            component="span"
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: 'block', mt: 1 }}
                          >
                            Path: {result.path}
                          </Typography>
                        </React.Fragment>
                      }
                    />
                  </ListItem>
                  <Divider component="li" />
                </React.Fragment>
              ))}
            </List>
          </Box>
        ) : searchTerm && !isSearching ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography>No results found for "{searchTerm}"</Typography>
          </Box>
        ) : null}
      </Paper>
    </Box>
  );
};

export default SearchComponent;
