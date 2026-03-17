import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { useApi } from '../../hooks/useApi';
import * as analyticsApi from '../../api/analytics';

const DOMAIN_NAMES = {
  dataCollection: 'Data Collection',
  assessment: 'Assessment',
  skillAcquisition: 'Skill Acquisition',
  behaviorReduction: 'Behavior Reduction',
  documentation: 'Documentation',
  crisisManagement: 'Crisis Management'
};

export const DomainProgressChart = () => {
  const { data, loading, error, execute } = useApi(analyticsApi.getDomainProgress);
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    execute();
  }, [execute]);

  useEffect(() => {
    if (data?.domainBreakdown) {
      const formatted = data.domainBreakdown.map(item => ({
        domain: DOMAIN_NAMES[item.domain] || item.domain,
        hours: item.qualifiedHours || 0,
        domainKey: item.domain
      }));
      setChartData(formatted);
    }
  }, [data]);

  if (loading) {
    return <div className="alert alert-info">Loading chart...</div>;
  }

  if (error) {
    return <div className="alert alert-danger">Error: {error}</div>;
  }

  if (chartData.length === 0) {
    return <div className="alert alert-info">No domain data available</div>;
  }

  return (
    <div className="chart-container">
      <h3>OJT Hours by Domain</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="domain"
            angle={-45}
            textAnchor="end"
            height={100}
          />
          <YAxis label={{ value: 'Qualified Hours', angle: -90, position: 'insideLeft' }} />
          <Tooltip
            formatter={(value) => `${value.toFixed(1)} hours`}
            labelFormatter={(label) => `${label}`}
          />
          <Legend />
          <Bar
            dataKey="hours"
            fill="#e84c1f"
            name="Qualified Hours"
            radius={[8, 8, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
