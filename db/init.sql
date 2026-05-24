-- PostgreSQL schema for the powerlifting video dataset
DROP TABLE IF EXISTS rep_results;
DROP TABLE IF EXISTS analysis_results;
DROP TABLE IF EXISTS videos;

CREATE TABLE videos (
    video_id VARCHAR(16) PRIMARY KEY,
    filename TEXT NOT NULL,
    exercise VARCHAR(20) NOT NULL,
    quality_label VARCHAR(20) NOT NULL,
    issue_label VARCHAR(40) NOT NULL,
    manual_comment TEXT,
    annotation_file TEXT NOT NULL,
    fps NUMERIC(8,3),
    total_frames INTEGER,
    duration_sec NUMERIC(8,3),
    width INTEGER,
    height INTEGER,
    split VARCHAR(16),
    source_type VARCHAR(40)
);

CREATE TABLE analysis_results (
    video_id VARCHAR(16) PRIMARY KEY REFERENCES videos(video_id),
    auto_exercise VARCHAR(20),
    program_score INTEGER,
    program_status TEXT,
    program_issues TEXT,
    reps_found INTEGER,
    pose_found_ratio NUMERIC(8,4),
    visibility NUMERIC(8,4),
    body_side VARCHAR(10)
);

CREATE TABLE rep_results (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(16) REFERENCES videos(video_id),
    rep_no INTEGER,
    start_sec NUMERIC(8,3),
    bottom_sec NUMERIC(8,3),
    end_sec NUMERIC(8,3),
    score INTEGER,
    status TEXT,
    issues TEXT,
    metrics_json JSONB
);
