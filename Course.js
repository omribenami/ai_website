const mongoose = require('mongoose');

const CourseSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Please provide a course title'],
    trim: true,
    maxlength: [100, 'Title cannot be more than 100 characters']
  },
  slug: {
    type: String,
    required: [true, 'Please provide a course slug'],
    unique: true,
    trim: true
  },
  description: {
    type: String,
    required: [true, 'Please provide a course description']
  },
  price: {
    type: Number,
    required: [true, 'Please provide a course price']
  },
  discountPrice: {
    type: Number
  },
  thumbnail: {
    type: String,
    default: 'default-course.jpg'
  },
  level: {
    type: String,
    enum: ['beginner', 'intermediate', 'advanced'],
    required: [true, 'Please specify course level']
  },
  prerequisites: [{
    type: String
  }],
  learningObjectives: [{
    type: String
  }],
  modules: [{
    title: {
      type: String,
      required: [true, 'Please provide a module title']
    },
    description: {
      type: String
    },
    order: {
      type: Number,
      required: [true, 'Please provide module order']
    },
    sections: [{
      title: {
        type: String,
        required: [true, 'Please provide a section title']
      },
      contentType: {
        type: String,
        enum: ['text', 'video', 'code', 'quiz'],
        required: [true, 'Please specify content type']
      },
      content: {
        type: String,
        required: [true, 'Please provide section content']
      },
      codeLanguage: {
        type: String
      },
      order: {
        type: Number,
        required: [true, 'Please provide section order']
      },
      duration: {
        type: Number // in minutes
      }
    }]
  }],
  quizzes: [{
    title: {
      type: String,
      required: [true, 'Please provide a quiz title']
    },
    description: {
      type: String
    },
    moduleId: {
      type: String,
      required: [true, 'Please specify which module this quiz belongs to']
    },
    questions: [{
      question: {
        type: String,
        required: [true, 'Please provide a question']
      },
      type: {
        type: String,
        enum: ['multiple-choice', 'true-false', 'coding'],
        required: [true, 'Please specify question type']
      },
      options: [{
        text: String,
        isCorrect: Boolean
      }],
      correctAnswer: {
        type: String
      },
      explanation: {
        type: String
      },
      points: {
        type: Number,
        default: 1
      }
    }]
  }],
  totalDuration: {
    type: Number // in minutes
  },
  totalModules: {
    type: Number
  },
  totalSections: {
    type: Number
  },
  ratings: [{
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    rating: {
      type: Number,
      min: 1,
      max: 5,
      required: [true, 'Please provide a rating']
    },
    review: {
      type: String
    },
    createdAt: {
      type: Date,
      default: Date.now
    }
  }],
  averageRating: {
    type: Number,
    min: [1, 'Rating must be at least 1'],
    max: [5, 'Rating cannot be more than 5']
  },
  numReviews: {
    type: Number,
    default: 0
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Calculate average rating when ratings are modified
CourseSchema.pre('save', function(next) {
  if (this.ratings && this.ratings.length > 0) {
    this.averageRating = this.ratings.reduce((acc, item) => item.rating + acc, 0) / this.ratings.length;
    this.numReviews = this.ratings.length;
  }
  
  // Calculate total modules and sections
  if (this.modules) {
    this.totalModules = this.modules.length;
    this.totalSections = this.modules.reduce((acc, module) => acc + (module.sections ? module.sections.length : 0), 0);
    
    // Calculate total duration
    this.totalDuration = this.modules.reduce((acc, module) => {
      return acc + module.sections.reduce((secAcc, section) => secAcc + (section.duration || 0), 0);
    }, 0);
  }
  
  next();
});

module.exports = mongoose.model('Course', CourseSchema);
