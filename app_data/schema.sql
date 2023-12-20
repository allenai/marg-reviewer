create table if not exists files (
	pdf_hash text not null,
	file_size integer not null,
	received_time timestamp not null,
	primary key (pdf_hash)
);

create table if not exists reviews (
	pdf_hash text not null,
	method text not null,
	review_json text not null,
	completed_time timestamp not null,
	primary key (pdf_hash, method)
);

create table if not exists surveys (
	pdf_hash text not null,
	user_id text not null,
	survey_json text not null,
	received_timestamp timestamp not null,
	ip text not null,
	user_agent text not null,
	primary key (pdf_hash, user_id, received_timestamp)
);

create table if not exists work_queue (
	pdf_hash text not null,
	status text not null,
	worker_pid integer,
	notification_email text not null,
	submitted_time timestamp not null,
	completed_time timestamp
);
