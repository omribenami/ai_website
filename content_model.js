// Content Model for AI Expert Guide Blog

// server/models/Content.js
const mongoose = require('mongoose');

const ContentSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Please add a title'],
    trim: true,
    maxlength: [100, 'Title cannot be more than 100 characters']
  },
  slug: {
    type: String,
    required: [true, 'Please add a slug'],
    unique: true,
    trim: true,
    lowercase: true
  },
  type: {
    type: String,
    required: [true, 'Please specify content type'],
    enum: ['module', 'lesson', 'quiz', 'exercise'],
    default: 'lesson'
  },
  content: {
    type: String,
    required: [true, 'Please add content']
  },
  accessLevel: {
    type: String,
    enum: ['free', 'premium'],
    default: 'premium'
  },
  parentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Content'
  },
  order: {
    type: Number,
    default: 0
  },
  createdDate: {
    type: Date,
    default: Date.now
  },
  updatedDate: {
    type: Date,
    default: Date.now
  },
  metadata: {
    readTime: Number,
    difficulty: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced'],
      default: 'beginner'
    },
    tags: [String]
  },
  resources: [
    {
      type: {
        type: String,
        enum: ['video', 'code', 'download'],
        default: 'code'
      },
      url: String,
      title: String
    }
  ]
});

// Create slug from title
ContentSchema.pre('save', function(next) {
  if (!this.isModified('title')) {
    next();
  }
  
  // Create slug from title if not provided
  if (!this.slug) {
    this.slug = this.title
      .toLowerCase()
      .replace(/[^\w ]+/g, '')
      .replace(/ +/g, '-');
  }
  
  // Update updatedDate on save
  this.updatedDate = Date.now();
  
  next();
});

module.exports = mongoose.model('Content', ContentSchema);
