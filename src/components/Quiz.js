import React, { useState } from 'react';
import { Box, Typography, Radio, RadioGroup, FormControlLabel, FormControl, Button, Paper, Divider, LinearProgress, Alert } from '@mui/material';
import CodeEditor from '../components/CodeEditor'; // ✅ correct


const Quiz = ({ 
  quiz = {
    id: 'sample-quiz',
    title: 'Sample Quiz',
    description: 'Test your knowledge with this quiz',
    questions: [
      {
        id: 'q1',
        question: 'What is the primary purpose of a neural network?',
        type: 'multiple-choice',
        options: [
          { id: 'a', text: 'To store data in a structured format' },
          { id: 'b', text: 'To create visual representations of data' },
          { id: 'c', text: 'To learn patterns from data and make predictions', isCorrect: true },
          { id: 'd', text: 'To optimize database queries' }
        ],
        explanation: 'Neural networks are designed to learn patterns from data and use those patterns to make predictions or decisions about new data.'
      },
      {
        id: 'q2',
        question: 'Which of the following is NOT a common activation function in neural networks?',
        type: 'multiple-choice',
        options: [
          { id: 'a', text: 'ReLU (Rectified Linear Unit)' },
          { id: 'b', text: 'Sigmoid' },
          { id: 'c', text: 'Tanh' },
          { id: 'd', text: 'Quadratic', isCorrect: true }
        ],
        explanation: 'Common activation functions include ReLU, Sigmoid, Tanh, and Softmax. Quadratic is not typically used as an activation function in neural networks.'
      },
      {
        id: 'q3',
        question: 'Write a Python function that creates a simple neural network layer with random weights.',
        type: 'coding',
        language: 'python',
        initialCode: 'import numpy as np\n\ndef create_layer(input_size, output_size):\n    # Your code here\n    pass\n\n# Test your function\nlayer = create_layer(3, 2)\nprint(layer)',
        solution: 'import numpy as np\n\ndef create_layer(input_size, output_size):\n    # Initialize random weights and biases\n    weights = np.random.randn(input_size, output_size)\n    biases = np.zeros(output_size)\n    return weights, biases\n\n# Test your function\nlayer = create_layer(3, 2)\nprint(layer)',
        explanation: 'This function creates a neural network layer by initializing random weights using numpy\'s random.randn function, which generates values from a standard normal distribution. The biases are initialized as zeros.'
      }
    ]
  },
  onComplete = () => {}
}) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [codeOutputs, setCodeOutputs] = useState({});
  
  const currentQuestion = quiz.questions[currentQuestionIndex];
  const totalQuestions = quiz.questions.length;
  
  const handleAnswerChange = (event) => {
    setAnswers({
      ...answers,
      [currentQuestion.id]: event.target.value
    });
  };
  
  const handleCodeChange = (code) => {
    setAnswers({
      ...answers,
      [currentQuestion.id]: code
    });
  };
  
  const handleCodeOutput = (questionId, output) => {
    setCodeOutputs({
      ...codeOutputs,
      [questionId]: output
    });
  };
  
  const handleNext = () => {
    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      setShowResults(true);
      onComplete(calculateScore());
    }
  };
  
  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };
  
  const calculateScore = () => {
    let correctAnswers = 0;
    
    quiz.questions.forEach(question => {
      if (question.type === 'multiple-choice') {
        const correctOption = question.options.find(option => option.isCorrect);
        if (correctOption && answers[question.id] === correctOption.id) {
          correctAnswers++;
        }
      } else if (question.type === 'coding') {
        // For coding questions, we would need a more sophisticated evaluation
        // For now, we'll just check if they submitted something
        if (answers[question.id] && answers[question.id] !== question.initialCode) {
          correctAnswers += 0.5; // Partial credit for attempting
        }
      }
    });
    
    return {
      score: correctAnswers,
      total: totalQuestions,
      percentage: Math.round((correctAnswers / totalQuestions) * 100)
    };
  };
  
  const isAnswered = (questionId) => {
    return answers[questionId] !== undefined && answers[questionId] !== '';
  };
  
  const renderQuestion = (question) => {
    switch (question.type) {
      case 'multiple-choice':
        return (
          <FormControl component="fieldset" sx={{ width: '100%' }}>
            <RadioGroup
              value={answers[question.id] || ''}
              onChange={handleAnswerChange}
            >
              {question.options.map(option => (
                <FormControlLabel
                  key={option.id}
                  value={option.id}
                  control={<Radio />}
                  label={option.text}
                  sx={{ mb: 1 }}
                  disabled={showResults}
                />
              ))}
            </RadioGroup>
          </FormControl>
        );
        
      case 'coding':
        return (
          <CodeEditor
            initialCode={answers[question.id] || question.initialCode}
            language={question.language || 'python'}
            onCodeChange={handleCodeChange}
            onRunCode={(output) => handleCodeOutput(question.id, output)}
            readOnly={showResults}
          />
        );
        
      default:
        return <Typography>Question type not supported</Typography>;
    }
  };
  
  const renderResults = () => {
    const score = calculateScore();
    
    return (
      <Box>
        <Typography variant="h5" gutterBottom>
          Quiz Results
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Score: {score.score} out of {score.total} ({score.percentage}%)
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={score.percentage} 
            sx={{ height: 10, borderRadius: 5 }}
          />
        </Box>
        
        {quiz.questions.map((question, index) => (
          <Paper key={question.id} sx={{ p: 3, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Question {index + 1}: {question.question}
            </Typography>
            
            {question.type === 'multiple-choice' && (
              <Box sx={{ mb: 2 }}>
                {question.options.map(option => {
                  const isSelected = answers[question.id] === option.id;
                  const isCorrect = option.isCorrect;
                  
                  return (
                    <Box 
                      key={option.id}
                      sx={{ 
                        p: 1, 
                        mb: 1, 
                        borderRadius: 1,
                        bgcolor: isSelected && isCorrect ? 'success.light' : 
                                 isSelected && !isCorrect ? 'error.light' : 
                                 !isSelected && isCorrect ? 'info.light' : 'transparent'
                      }}
                    >
                      <Typography>
                        {option.text}
                        {isSelected && isCorrect && ' ✓'}
                        {isSelected && !isCorrect && ' ✗'}
                        {!isSelected && isCorrect && ' (Correct Answer)'}
                      </Typography>
                    </Box>
                  );
                })}
              </Box>
            )}
            
            {question.type === 'coding' && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Your Solution:
                </Typography>
                <CodeEditor
                  initialCode={answers[question.id] || question.initialCode}
                  language={question.language || 'python'}
                  readOnly={true}
                  height="200px"
                />
                
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Example Solution:
                </Typography>
                <CodeEditor
                  initialCode={question.solution}
                  language={question.language || 'python'}
                  readOnly={true}
                  height="200px"
                />
              </Box>
            )}
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle2" gutterBottom>
              Explanation:
            </Typography>
            <Typography variant="body2">
              {question.explanation}
            </Typography>
          </Paper>
        ))}
        
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Button 
            variant="contained" 
            color="primary"
            onClick={() => {
              setCurrentQuestionIndex(0);
              setShowResults(false);
              setAnswers({});
              setCodeOutputs({});
            }}
          >
            Retake Quiz
          </Button>
        </Box>
      </Box>
    );
  };
  
  if (showResults) {
    return renderResults();
  }
  
  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        {quiz.title}
      </Typography>
      
      <Typography variant="body2" color="text.secondary" gutterBottom>
        {quiz.description}
      </Typography>
      
      <LinearProgress 
        variant="determinate" 
        value={(currentQuestionIndex / totalQuestions) * 100} 
        sx={{ height: 6, borderRadius: 3, mb: 3 }}
      />
      
      <Typography variant="subtitle1" gutterBottom>
        Question {currentQuestionIndex + 1} of {totalQuestions}
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {currentQuestion.question}
        </Typography>
        
        {renderQuestion(currentQuestion)}
      </Paper>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button 
          variant="outlined" 
          onClick={handlePrevious}
          disabled={currentQuestionIndex === 0}
        >
          Previous
        </Button>
        
        <Button 
          variant="contained" 
          color="primary"
          onClick={handleNext}
          disabled={!isAnswered(currentQuestion.id)}
        >
          {currentQuestionIndex < totalQuestions - 1 ? 'Next' : 'Finish Quiz'}
        </Button>
      </Box>
    </Box>
  );
};

export default Quiz;
