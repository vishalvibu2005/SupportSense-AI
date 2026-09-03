import sqlite3

import pandas as pd
import streamlit as st

from ai_engine import (
    assess_sla_risk,
    find_similar_resolved_tickets,
    get_suggested_resolution,
    predict_category,
    predict_priority,
)   

DB_NAME = "supportsense.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    columns = [
        row[1]
        for row in cursor.execute("PRAGMA table_info(tickets)").fetchall()
    ]

    new_columns = {
    "predicted_category": "TEXT",
    "predicted_priority": "TEXT",
    "sla_risk": "TEXT",
    "resolution_note": "TEXT",
    }

    for column_name, column_type in new_columns.items():
        if column_name not in columns:
            cursor.execute(
                f"ALTER TABLE tickets ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()
    connection.close()


def add_ticket(issue_title, description, category, priority, employee_name):
    ticket_text = f"{issue_title} {description}"

    predicted_category, _ = predict_category(ticket_text)
    predicted_priority, _ = predict_priority(ticket_text)
    sla_risk, _ = assess_sla_risk(
        ticket_text,
        predicted_priority
    )

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO tickets (
            issue_title,
            description,
            category,
            priority,
            employee_name,
            predicted_category,
            predicted_priority,
            sla_risk
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        issue_title,
        description,
        category,
        priority,
        employee_name,
        predicted_category,
        predicted_priority,
        sla_risk,
    ))

    connection.commit()
    connection.close()

    return predicted_category, predicted_priority, sla_risk


def get_tickets():
    connection = get_connection()

    tickets = pd.read_sql_query("""
        SELECT
            ticket_id,
            issue_title,
            description,
            category,
            priority,
            employee_name,
            status,
            predicted_category,
            predicted_priority,
            sla_risk,
            resolution_note,
            created_at
        FROM tickets
        ORDER BY ticket_id DESC
    """, connection)

    connection.close()
    return tickets


def update_ticket_status(ticket_id, new_status, resolution_note):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE tickets
        SET status = ?, resolution_note = ?
        WHERE ticket_id = ?
    """, (
        new_status,
        resolution_note,
        ticket_id,
    ))

    connection.commit()
    connection.close()

def prepare_display_table(dataframe):
    display_tickets = dataframe.copy()

    display_tickets["ticket_id"] = display_tickets["ticket_id"].apply(
        lambda ticket_id: f"INC-{ticket_id + 1000}"
    )

    return display_tickets.rename(
        columns={
            "ticket_id": "Ticket ID",
            "issue_title": "Issue",
            "category": "Selected Category",
            "priority": "Selected Priority",
            "employee_name": "Employee",
            "status": "Status",
            "predicted_category": "AI Category",
            "predicted_priority": "AI Priority",
            "sla_risk": "SLA Risk",
            "created_at": "Created At",
            "resolution_note": "Resolution Note",
        }
    )


st.set_page_config(
    page_title="SupportSense AI",
    page_icon="🛠️",
    layout="wide",
)

initialize_database()

st.sidebar.title("🛠️ SupportSense AI")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Ticket Center"],
)

tickets = get_tickets()

if page == "Dashboard":
    st.title("Operations Dashboard")
    st.caption("Monitor support workload, priorities, and SLA risks.")

    total_tickets = len(tickets)
    open_tickets = len(tickets[tickets["status"] == "Open"])
    high_risk_tickets = len(
        tickets[tickets["sla_risk"] == "High"]
    )

    column1, column2, column3 = st.columns(3)
    column1.metric("Total Tickets", total_tickets)
    column2.metric("Open Tickets", open_tickets)
    column3.metric("High SLA-Risk Tickets", high_risk_tickets)

    st.divider()

    if tickets.empty:
        st.info("No tickets have been created yet.")
    else:
        left_column, right_column = st.columns(2)

        with left_column:
            st.subheader("Tickets by Category")
            st.bar_chart(tickets["category"].value_counts())

        with right_column:
            st.subheader("Tickets by SLA Risk")
            risk_data = tickets["sla_risk"].fillna("Not Analyzed")
            st.bar_chart(risk_data.value_counts())

        st.subheader("Recent Support Tickets")

        recent_columns = [
            "ticket_id",
            "issue_title",
            "category",
            "priority",
            "predicted_category",
            "predicted_priority",
            "sla_risk",
            "status",
            "created_at",
        ]

        st.dataframe(
            prepare_display_table(tickets[recent_columns]),
            width="stretch",
            hide_index=True,
            
        )

if page == "Ticket Center":
    st.title("Ticket Center")
    st.caption("Create, search, filter, and update support tickets.")

    create_tab, manage_tab = st.tabs(
        ["Create Ticket", "Manage Tickets"]
    )

    with create_tab:
        st.subheader("Create a Support Ticket")

        st.markdown("### 🤖 AI Ticket Analyzer")

        analysis_text = st.text_area(
            "Describe an IT issue for AI analysis",
            placeholder="Example: I cannot connect to VPN from home.",
            key="ai_analysis_text",
        )

        if st.button("Analyze with AI", key="analyze_button"):
            if not analysis_text.strip():
                st.warning("Please describe an issue first.")
            else:
                predicted_category, category_confidence = predict_category(
                    analysis_text
                )

                predicted_priority, priority_confidence = predict_priority(
                    analysis_text
                )

                sla_risk, sla_reason = assess_sla_risk(
                    analysis_text,
                    predicted_priority,
                )

                category_column, priority_column, sla_column = st.columns(3)

                with category_column:
                    st.success(
                        f"Predicted category: {predicted_category}"
                    )

                with priority_column:
                    st.warning(
                        f"Predicted priority: {predicted_priority}"
                    )

                with sla_column:
                    if sla_risk == "High":
                        st.error(f"SLA risk: {sla_risk}")
                    elif sla_risk == "Medium":
                        st.warning(f"SLA risk: {sla_risk}")
                    else:
                        st.info(f"SLA risk: {sla_risk}")

                st.caption(
                    f"Category confidence: {category_confidence:.0%} "
                    f"| Priority confidence: {priority_confidence:.0%}"
                )

                st.caption(f"SLA note: {sla_reason}")
                suggested_steps = get_suggested_resolution(
                    predicted_category,
                    predicted_priority,
                )

                with st.expander("Suggested resolution steps"):
                    for step in suggested_steps:
                        st.write(f"• {step}")
        st.divider()
        st.markdown("### Create Ticket Record")
        similar_tickets = find_similar_resolved_tickets(
                    analysis_text,
                    tickets.to_dict("records"),
                )

        with st.expander("Similar past resolved tickets"):
                    if similar_tickets:
                        for ticket in similar_tickets:
                            ticket_number = ticket["ticket_id"] + 1000

                            st.markdown(
                                f"**INC-{ticket_number}: "
                                f"{ticket['issue_title']}**"
                            )

                            st.write(
                                f"Similarity: "
                                f"{ticket['similarity_score']:.0%}"
                            )

                            st.write(
                                f"Past resolution: "
                                f"{ticket['resolution_note']}"
                            )

                            st.divider()
                    else:
                        st.info(
                            "No similar resolved tickets were found yet."
                        )

        with st.form("ticket_form", clear_on_submit=True):
            issue_title = st.text_input(
                "Issue title",
                placeholder="Example: Unable to connect to company VPN",
            )


            description = st.text_area(
                "Describe the issue",
                placeholder="Explain what happened and when it started.",
            )

            column1, column2 = st.columns(2)

            with column1:
                category = st.selectbox(
                    "Select category",
                    ["Access", "Network", "Hardware", "Software", "Other"],
                )

            with column2:
                priority = st.selectbox(
                    "Select priority",
                    ["Low", "Medium", "High", "Critical"],
                )

            employee_name = st.text_input(
                "Employee name",
                placeholder="Example: Vishal",
            )

            submitted = st.form_submit_button("Create Ticket")

            if submitted:
                if not issue_title or not description or not employee_name:
                    st.error(
                        "Please fill in the issue title, description, "
                        "and employee name."
                    )
                else:
                    ai_category, ai_priority, sla_risk = add_ticket(
                        issue_title,
                        description,
                        category,
                        priority,
                        employee_name,
                    )

                    st.success("Ticket created and AI assessment saved!")

                    st.caption(
                        f"Saved AI assessment: {ai_category} | "
                        f"{ai_priority} priority | "
                        f"{sla_risk} SLA risk"
                    )

    with manage_tab:
        st.subheader("Find and Manage Tickets")

        if tickets.empty:
            st.info("Create a ticket first.")
        else:
            filter_column1, filter_column2, filter_column3 = st.columns(3)

            with filter_column1:
                selected_category = st.selectbox(
                    "Filter by category",
                    ["All"] + sorted(tickets["category"].unique().tolist()),
                )

            with filter_column2:
                selected_priority = st.selectbox(
                    "Filter by priority",
                    ["All"] + sorted(tickets["priority"].unique().tolist()),
                )

            with filter_column3:
                selected_status = st.selectbox(
                    "Filter by status",
                    ["All"] + sorted(tickets["status"].unique().tolist()),
                )

            filtered_tickets = tickets.copy()

            if selected_category != "All":
                filtered_tickets = filtered_tickets[
                    filtered_tickets["category"] == selected_category
                ]

            if selected_priority != "All":
                filtered_tickets = filtered_tickets[
                    filtered_tickets["priority"] == selected_priority
                ]

            if selected_status != "All":
                filtered_tickets = filtered_tickets[
                    filtered_tickets["status"] == selected_status
                ]

            search_text = st.text_input(
                "Search by issue title or employee name"
            )

            if search_text:
                search_text = search_text.lower()

                filtered_tickets = filtered_tickets[
                    filtered_tickets["issue_title"].str.lower().str.contains(
                        search_text,
                        na=False,
                    )
                    | filtered_tickets["employee_name"].str.lower().str.contains(
                        search_text,
                        na=False,
                    )
                ]

            table_columns = [
                "ticket_id",
                "issue_title",
                "category",
                "priority",
                "employee_name",
                "status",
                "predicted_category",
                "predicted_priority",
                "sla_risk",
                "resolution_note",
                "created_at",
            ]

            st.dataframe(
                prepare_display_table(filtered_tickets[table_columns]),
                width="stretch",
                hide_index=True,
            )
            export_table = prepare_display_table(
                filtered_tickets[table_columns]
            )

            csv_report = export_table.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Filtered Ticket Report (CSV)",
                data=csv_report,
                file_name="supportsense_ticket_report.csv",
                mime="text/csv",
            )
            st.divider()
            st.subheader("Update Ticket Status")

            selected_ticket_id = st.selectbox(
                "Select a ticket",
                tickets["ticket_id"].tolist(),
                format_func=lambda ticket_id: (
                    f"INC-{ticket_id + 1000} — "
                    f"{tickets.loc[tickets['ticket_id'] == ticket_id, 'issue_title'].iloc[0]}"
                ),
            )

            current_status = tickets.loc[
                tickets["ticket_id"] == selected_ticket_id,
                "status",
            ].iloc[0]

            status_options = ["Open", "In Progress", "Resolved"]

            new_status = st.selectbox(
                "New status",
                status_options,
                index=status_options.index(current_status),
            )

            current_resolution = tickets.loc[
                tickets["ticket_id"] == selected_ticket_id,
                "resolution_note",
            ].iloc[0]

            if pd.isna(current_resolution):
                current_resolution = ""

            resolution_note = st.text_area(
                "Resolution note",
                value=current_resolution,
                placeholder=(
                    "Example: Reset the password and confirmed that "
                    "the employee could log in."
                ),
            )

            if st.button("Update Status"):
                if new_status == "Resolved" and not resolution_note.strip():
                    st.error(
                        "Please add a resolution note before resolving "
                        "the ticket."
                    )
                else:
                    update_ticket_status(
                        selected_ticket_id,
                        new_status,
                        resolution_note,
                    )
                    st.success("Ticket status updated successfully!")
                    st.rerun()