import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Humanoid Robotics',
    Svg: require('../../static/img/robot-arm.svg').default,
    description: (
      <>
        Learn the fundamentals of Physical AI and Humanoid Robotics through
        interactive modules and hands-on projects.
      </>
    ),
  },
  {
    title: 'Four Comprehensive Modules',
    Svg: require('../../static/img/digital-twin.svg').default,
    description: (
      <>
        Master ROS 2, Digital Twins, AI-Robot Brains, and Vision-Language-Action systems
        through structured learning paths.
      </>
    ),
  },
  {
    title: 'Interactive Learning',
    Svg: require('../../static/img/interactive-learning.svg').default,
    description: (
      <>
        Engage with RAG chatbots, code examples, simulation viewers, and assessments
        to reinforce your learning.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}