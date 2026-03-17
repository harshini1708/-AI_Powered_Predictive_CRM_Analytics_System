import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { useApi } from '../../hooks/useApi';
import * as analyticsApi from '../../api/analytics';

const DOMAIN_LABELS = {
  A: 'Professional Conduct',
  B: 'Assessment',
  C: 'Skill Acquisition',
  D: 'Behavior Reduction',
  E: 'Documentation',
  F: 'Professional Development'
};

const SCORE_COLORS = {
  1: '#dc3545', // Red - Not Yet
  2: '#ff9800', // Orange - Emerging
  3: '#ffc107', // Yellow - Developing
  4: '#28a745', // Green - Proficient
  5: '#007bff'  // Blue - Mastery
};

const SCORE_LABELS = {
  1: 'Not Yet',
  2: 'Emerging',
  3: 'Developing',
  4: 'Proficient',
  5: 'Mastery'
};

export const CompetencyHeatmap = () => {
  const { data, loading, error, execute } = useApi(analyticsApi.getCompetencyHeatMap);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    execute();
  }, [execute]);

  useEffect(() => {
    if (data?.competencyMap) {
      const formatted = data.competencyMap.map(item => ({
        domain: DOMAIN_LABELS[item.domain] || item.domain,
        domainKey: item.domain,
        avgScore: parseFloat(item.avgScore) || 0,
        count: item.count || 0
      }));
      setChartData(formatted);
    }
  }, [data]);

  if (loading) {
    return <div className="alert alert-info">Loading heatmap...</div>;
  }

  if (error) {
    return <div className="alert alert-danger">Error: {error}</div>;
  }

  if (chartData.length === 0) {
    return <div className="alert alert-info">No competency data available</div>;
  }

  const getColorByScore = (score) => {
    const roundedScore = Math.round(score);
    return SCORE_COLORS[roundedScore] || '#6c757d';
  };

  return (
    <div className="chart-container">
      <h3>Competency Ratings by Domain (Average Score)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="domain" />
          <YAxis
            domain={[0, 5]}
            label={{ value: 'Average Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            formatter={(value) => {
              const rounded = Math.round(value);
              return [`${value.toFixed(2)} (${SCORE_LABELS[rounded] || 'Unknown'})`, 'Score'];
            }}
            labelFormatter={(label) => `${label}`}
          />
          <Legend />
          <Bar dataKey="avgScore" name="Average Competency Score" radius={[8, 8, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColorByScore(entry.avgScore)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="chart-legend" style={{ marginTop: '20px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
          {[1, 2, 3, 4, 5].map(score => (
            <div
              key={score}
              style={{
                padding: '10px',
                backgroundColor: SCORE_COLORS[score],
                color: 'white',
                borderRadius: '4px',
                textAlign: 'center',
                fontSize: '12px',
                fontWeight: '600'
              }}
            >
              {score} - {SCORE_LABELS[score]}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
