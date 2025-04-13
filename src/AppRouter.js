import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import CourseCatalog from './pages/CourseCatalog';
import CourseDetails from './pages/CourseDetails';
import CourseViewer from './pages/CourseViewer';
import TestPage from './pages/TestPage';

// Import other pages as needed

function AppRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/courses" element={<CourseCatalog />} />
        <Route path="/courses/:courseId" element={<CourseDetails />} />
        <Route path="/learn/:courseId" element={<CourseViewer />} />
        <Route path="/test" element={<TestPage />} />
        {/* Add other routes as needed */}
      </Routes>
    </Router>
  );
}

export default AppRouter;
