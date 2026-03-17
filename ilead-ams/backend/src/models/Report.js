import mongoose from 'mongoose';

const reportSchema = new mongoose.Schema(
  {
    // Basic Info
    name: {
      type: String,
      required: true,
      index: true
    },

    description: String,

    type: {
      type: String,
      enum: [
        'hours_summary',
        'apprentice_progress',
        'supervisor_workload',
        'program_statistics',
        'completion_forecast',
        'custom'
      ],
      default: 'custom'
    },

    // Creator and Ownership
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true
    },

    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true
    },

    // Configuration
    filters: {
      dateRange: {
        startDate: Date,
        endDate: Date
      },
      apprenticeIds: [mongoose.Schema.Types.ObjectId],
      supervisorIds: [mongoose.Schema.Types.ObjectId],
      programs: [String],
      status: [String]
    },

    // Columns/Metrics
    columns: [{
      name: String,
      type: {
        type: String,
        enum: ['text', 'number', 'percentage', 'date', 'currency']
      },
      label: String,
      aggregation: {
        type: String,
        enum: ['sum', 'avg', 'count', 'min', 'max', 'none'],
        default: 'none'
      }
    }],

    // Grouping
    groupBy: {
      field: String,
      type: {
        type: String,
        enum: ['apprentice', 'program', 'supervisor', 'date', 'status', 'none'],
        default: 'none'
      }
    },

    // Sorting
    sortBy: {
      field: String,
      order: {
        type: String,
        enum: ['asc', 'desc'],
        default: 'asc'
      }
    },

    // Scheduling
    schedule: {
      enabled: {
        type: Boolean,
        default: false
      },

      frequency: {
        type: String,
        enum: ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'once'],
        default: 'weekly'
      },

      dayOfWeek: Number, // 0-6, Monday=1
      dayOfMonth: Number, // 1-31
      time: String, // HH:mm format
      timezone: {
        type: String,
        default: 'UTC'
      },

      nextRun: Date,
      lastRun: Date
    },

    // Delivery
    delivery: {
      enabled: {
        type: Boolean,
        default: false
      },

      method: {
        type: String,
        enum: ['email', 'webhook', 'download'],
        default: 'email'
      },

      recipients: [
        {
          type: String, // email address or webhook URL
          format: String // 'pdf', 'excel', 'csv'
        }
      ],

      includeChart: {
        type: Boolean,
        default: true
      }
    },

    // Display Options
    displayOptions: {
      showChart: {
        type: Boolean,
        default: true
      },

      chartType: {
        type: String,
        enum: ['bar', 'line', 'pie', 'area', 'table'],
        default: 'bar'
      },

      pageSize: {
        type: Number,
        default: 50
      },

      showTotals: {
        type: Boolean,
        default: true
      },

      showPagination: {
        type: Boolean,
        default: true
      }
    },

    // Status and Tracking
    status: {
      type: String,
      enum: ['draft', 'active', 'archived'],
      default: 'draft'
    },

    // Execution History
    executions: [
      {
        executedAt: Date,
        executedBy: mongoose.Schema.Types.ObjectId,
        rowCount: Number,
        generatedFile: String,
        deliveryStatus: String,
        error: String
      }
    ],

    // Last Generated Report
    lastGeneratedAt: Date,
    lastGeneratedFile: String,
    lastRowCount: Number,

    // Sharing
    isPublic: {
      type: Boolean,
      default: false
    },

    sharedWith: [
      {
        userId: mongoose.Schema.Types.ObjectId,
        permission: {
          type: String,
          enum: ['view', 'edit', 'admin'],
          default: 'view'
        }
      }
    ],

    // Tags for organization
    tags: [String],

    // Metadata
    metadata: mongoose.Schema.Types.Mixed
  },
  {
    timestamps: true
  }
);

// Indexes
reportSchema.index({ createdBy: 1, status: 1 });
reportSchema.index({ owner: 1, createdAt: -1 });
reportSchema.index({ type: 1, status: 1 });
reportSchema.index({ 'schedule.enabled': 1, 'schedule.nextRun': 1 });
reportSchema.index({ isPublic: 1, createdAt: -1 });

// Pre-save middleware to set nextRun
reportSchema.pre('save', function(next) {
  if (this.schedule.enabled && !this.schedule.nextRun) {
    // Calculate next run time based on schedule
    this.schedule.nextRun = calculateNextRun(this.schedule);
  }
  next();
});

/**
 * Calculate next run time based on schedule
 */
function calculateNextRun(schedule) {
  const now = new Date();
  const next = new Date(now);

  switch (schedule.frequency) {
    case 'daily':
      next.setDate(next.getDate() + 1);
      break;
    case 'weekly':
      next.setDate(next.getDate() + (7 - next.getDay() + (schedule.dayOfWeek || 1)));
      break;
    case 'monthly':
      next.setMonth(next.getMonth() + 1);
      next.setDate(schedule.dayOfMonth || 1);
      break;
    case 'quarterly':
      next.setMonth(next.getMonth() + 3);
      break;
    case 'yearly':
      next.setFullYear(next.getFullYear() + 1);
      break;
    case 'once':
      return null;
  }

  // Set time
  if (schedule.time) {
    const [hours, minutes] = schedule.time.split(':').map(Number);
    next.setHours(hours, minutes, 0, 0);
  }

  return next;
}

export default mongoose.model('Report', reportSchema);
