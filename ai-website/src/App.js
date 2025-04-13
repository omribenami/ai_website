import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import pages (to be created)
import HomePage from './pages/HomePage';
import CourseCatalog from './pages/CourseCatalog';
import CourseDetails from './pages/CourseDetails';
import AboutPage from './pages/AboutPage';
import ContactPage from './pages/ContactPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import Dashboard from './pages/Dashboard';
import MyCourses from './pages/MyCourses';
import CourseViewer from './pages/CourseViewer';
import ProfilePage from './pages/ProfilePage';
import CheckoutPage from './pages/CheckoutPage';
import AdminDashboard from './pages/AdminDashboard';
import NotFoundPage from './pages/NotFoundPage';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#3498db', // Blue
    },
    secondary: {
      main: '#2ecc71', // Green
    },
    error: {
      main: '#e74c3c', // Red
    },
    warning: {
      main: '#f39c12', // Orange
    },
    info: {
      main: '#3498db', // Blue
    },
    success: {
      main: '#2ecc71', // Green
    },
    background: {
      default: '#ecf0f1', // Light Gray
      paper: '#ffffff', // White
    },
    text: {
      primary: '#2c3e50', // Dark Blue
      secondary: '#7f8c8d', // Gray
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 700,
    },
    h2: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 700,
    },
    h3: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 600,
    },
    h4: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 600,
    },
    h5: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 500,
    },
    h6: {
      fontFamily: 'Poppins, sans-serif',
      fontWeight: 500,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<HomePage />} />
          <Route path="/courses" element={<CourseCatalog />} />
          <Route path="/courses/:courseId" element={<CourseDetails />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          
          {/* Protected routes (to be implemented with auth) */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/my-courses" element={<MyCourses />} />
          <Route path="/learn/:courseId" element={<CourseViewer />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/checkout" element={<CheckoutPage />} />
          
          {/* Admin routes (to be implemented with auth) */}
          <Route path="/admin/*" element={<AdminDashboard />} />
          
          {/* 404 route */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
