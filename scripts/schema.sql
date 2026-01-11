--
-- PostgreSQL database dump
--

\restrict yARTxaIiCwtGX6Vm6h7Oempr62PnuiwMc2O2tv7vzeA8XzcQwqzVLDYQJ8h9npe

-- Dumped from database version 16.11
-- Dumped by pg_dump version 18.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: context_strategies; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.context_strategies (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    message_id uuid NOT NULL,
    strategy jsonb NOT NULL,
    prompt_key text NOT NULL,
    prompt_version integer NOT NULL,
    rendered_prompt text,
    raw_response text,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.context_strategies OWNER TO lattice;

--
-- Name: dreaming_proposals; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.dreaming_proposals (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    proposal_type text DEFAULT 'prompt_optimization'::text NOT NULL,
    prompt_key text,
    current_version integer,
    proposed_version integer,
    current_template text,
    proposed_template text,
    rationale text,
    proposal_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    review_data jsonb,
    applied_changes jsonb DEFAULT '[]'::jsonb,
    status text DEFAULT 'pending'::text,
    human_feedback text,
    rendered_optimization_prompt text,
    created_at timestamp with time zone DEFAULT now(),
    reviewed_at timestamp with time zone,
    reviewed_by text,
    CONSTRAINT dreaming_proposals_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'approved'::text, 'rejected'::text, 'deferred'::text])))
);


ALTER TABLE public.dreaming_proposals OWNER TO lattice;

--
-- Name: entities; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.entities (
    name text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.entities OWNER TO lattice;

--
-- Name: predicates; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.predicates (
    name text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.predicates OWNER TO lattice;

--
-- Name: prompt_audits; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.prompt_audits (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    prompt_key text NOT NULL,
    template_version integer,
    message_id uuid,
    rendered_prompt text NOT NULL,
    response_content text NOT NULL,
    model text,
    provider text,
    prompt_tokens integer,
    completion_tokens integer,
    cost_usd numeric(10,6),
    latency_ms integer,
    context_config jsonb,
    archetype_matched text,
    archetype_confidence double precision,
    reasoning jsonb,
    main_discord_message_id bigint,
    dream_discord_message_id bigint,
    feedback_id uuid,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT prompt_audits_rendered_prompt_check CHECK ((length(rendered_prompt) <= 50000))
);


ALTER TABLE public.prompt_audits OWNER TO lattice;

--
-- Name: prompt_registry; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.prompt_registry (
    prompt_key text NOT NULL,
    version integer NOT NULL,
    template text NOT NULL,
    temperature double precision DEFAULT 0.2,
    active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.prompt_registry OWNER TO lattice;

--
-- Name: raw_messages; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.raw_messages (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    discord_message_id bigint NOT NULL,
    channel_id bigint NOT NULL,
    content text NOT NULL,
    is_bot boolean DEFAULT false,
    is_proactive boolean DEFAULT false,
    generation_metadata jsonb,
    "timestamp" timestamp with time zone DEFAULT now(),
    user_timezone text
);


ALTER TABLE public.raw_messages OWNER TO lattice;

--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.schema_migrations (
    version integer NOT NULL,
    name text NOT NULL,
    applied_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.schema_migrations OWNER TO lattice;

--
-- Name: semantic_memories; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.semantic_memories (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,
    source_batch_id text,
    superseded_by uuid,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.semantic_memories OWNER TO lattice;

--
-- Name: system_health; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.system_health (
    metric_key text NOT NULL,
    metric_value text,
    recorded_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.system_health OWNER TO lattice;

--
-- Name: user_feedback; Type: TABLE; Schema: public; Owner: lattice
--

CREATE TABLE public.user_feedback (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    content text,
    sentiment text,
    referenced_discord_message_id bigint,
    user_discord_message_id bigint,
    audit_id uuid,
    strategy_id uuid,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT user_feedback_sentiment_check CHECK ((sentiment = ANY (ARRAY['positive'::text, 'negative'::text, 'neutral'::text])))
);


ALTER TABLE public.user_feedback OWNER TO lattice;

--
-- Name: context_strategies context_strategies_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.context_strategies
    ADD CONSTRAINT context_strategies_pkey PRIMARY KEY (id);


--
-- Name: dreaming_proposals dreaming_proposals_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.dreaming_proposals
    ADD CONSTRAINT dreaming_proposals_pkey PRIMARY KEY (id);


--
-- Name: entities entities_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (name);


--
-- Name: predicates predicates_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.predicates
    ADD CONSTRAINT predicates_pkey PRIMARY KEY (name);


--
-- Name: prompt_audits prompt_audits_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.prompt_audits
    ADD CONSTRAINT prompt_audits_pkey PRIMARY KEY (id);


--
-- Name: prompt_registry prompt_registry_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.prompt_registry
    ADD CONSTRAINT prompt_registry_pkey PRIMARY KEY (prompt_key, version);


--
-- Name: raw_messages raw_messages_discord_message_id_key; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.raw_messages
    ADD CONSTRAINT raw_messages_discord_message_id_key UNIQUE (discord_message_id);


--
-- Name: raw_messages raw_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.raw_messages
    ADD CONSTRAINT raw_messages_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: semantic_memories semantic_memories_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.semantic_memories
    ADD CONSTRAINT semantic_memories_pkey PRIMARY KEY (id);


--
-- Name: system_health system_health_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.system_health
    ADD CONSTRAINT system_health_pkey PRIMARY KEY (metric_key);


--
-- Name: user_feedback user_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.user_feedback
    ADD CONSTRAINT user_feedback_pkey PRIMARY KEY (id);


--
-- Name: idx_audits_created; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_audits_created ON public.prompt_audits USING btree (created_at DESC);


--
-- Name: idx_audits_dream_message; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_audits_dream_message ON public.prompt_audits USING btree (dream_discord_message_id);


--
-- Name: idx_audits_feedback; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_audits_feedback ON public.prompt_audits USING btree (feedback_id) WHERE (feedback_id IS NOT NULL);


--
-- Name: idx_audits_main_message; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_audits_main_message ON public.prompt_audits USING btree (main_discord_message_id);


--
-- Name: idx_audits_prompt_key; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_audits_prompt_key ON public.prompt_audits USING btree (prompt_key);


--
-- Name: idx_dreaming_proposals_created_at; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_dreaming_proposals_created_at ON public.dreaming_proposals USING btree (created_at DESC);


--
-- Name: idx_dreaming_proposals_prompt_key; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_dreaming_proposals_prompt_key ON public.dreaming_proposals USING btree (prompt_key);


--
-- Name: idx_dreaming_proposals_status; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_dreaming_proposals_status ON public.dreaming_proposals USING btree (status) WHERE (status = 'pending'::text);


--
-- Name: idx_dreaming_proposals_type_status; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_dreaming_proposals_type_status ON public.dreaming_proposals USING btree (proposal_type, status) WHERE (status = 'pending'::text);


--
-- Name: idx_entities_created_at; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_entities_created_at ON public.entities USING btree (created_at DESC);


--
-- Name: idx_predicates_created_at; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_predicates_created_at ON public.predicates USING btree (created_at DESC);


--
-- Name: idx_raw_messages_discord_id; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_raw_messages_discord_id ON public.raw_messages USING btree (discord_message_id);


--
-- Name: idx_raw_messages_timestamp; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_raw_messages_timestamp ON public.raw_messages USING btree ("timestamp" DESC);


--
-- Name: idx_schema_migrations_version; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_schema_migrations_version ON public.schema_migrations USING btree (version);


--
-- Name: idx_semantic_memories_active; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_active ON public.semantic_memories USING btree (subject, predicate, created_at DESC) WHERE (superseded_by IS NULL);


--
-- Name: idx_semantic_memories_created_at; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_created_at ON public.semantic_memories USING btree (created_at DESC);


--
-- Name: idx_semantic_memories_object; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_object ON public.semantic_memories USING btree (object);


--
-- Name: idx_semantic_memories_predicate; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_predicate ON public.semantic_memories USING btree (predicate);


--
-- Name: idx_semantic_memories_source_batch; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_source_batch ON public.semantic_memories USING btree (source_batch_id);


--
-- Name: idx_semantic_memories_subject; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_semantic_memories_subject ON public.semantic_memories USING btree (subject);


--
-- Name: idx_strategies_created_at; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_strategies_created_at ON public.context_strategies USING btree (created_at DESC);


--
-- Name: idx_strategies_entities; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_strategies_entities ON public.context_strategies USING gin (((strategy -> 'entities'::text)));


--
-- Name: idx_user_feedback_audit_id; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_user_feedback_audit_id ON public.user_feedback USING btree (audit_id) WHERE (audit_id IS NOT NULL);


--
-- Name: idx_user_feedback_sentiment; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_user_feedback_sentiment ON public.user_feedback USING btree (sentiment) WHERE (sentiment IS NOT NULL);


--
-- Name: idx_user_feedback_strategy_id; Type: INDEX; Schema: public; Owner: lattice
--

CREATE INDEX idx_user_feedback_strategy_id ON public.user_feedback USING btree (strategy_id) WHERE (strategy_id IS NOT NULL);


--
-- Name: context_strategies context_strategies_message_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.context_strategies
    ADD CONSTRAINT context_strategies_message_id_fkey FOREIGN KEY (message_id) REFERENCES public.raw_messages(id) ON DELETE CASCADE;


--
-- Name: prompt_audits prompt_audits_message_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.prompt_audits
    ADD CONSTRAINT prompt_audits_message_id_fkey FOREIGN KEY (message_id) REFERENCES public.raw_messages(id);


--
-- Name: semantic_memories semantic_memories_superseded_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.semantic_memories
    ADD CONSTRAINT semantic_memories_superseded_by_fkey FOREIGN KEY (superseded_by) REFERENCES public.semantic_memories(id);


--
-- Name: user_feedback user_feedback_audit_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.user_feedback
    ADD CONSTRAINT user_feedback_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES public.prompt_audits(id);


--
-- Name: user_feedback user_feedback_strategy_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: lattice
--

ALTER TABLE ONLY public.user_feedback
    ADD CONSTRAINT user_feedback_strategy_id_fkey FOREIGN KEY (strategy_id) REFERENCES public.context_strategies(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict yARTxaIiCwtGX6Vm6h7Oempr62PnuiwMc2O2tv7vzeA8XzcQwqzVLDYQJ8h9npe

