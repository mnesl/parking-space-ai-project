# AeroPark Smart Garage — Booking Process, Policies & FAQ

## 1. How the Interactive Reservation Process Works
Reserving a parking space through the AeroPark AI Chatbot is a fast, 3-step process:

### Step 1: Provide Your Reservation Details to the Chatbot
To initiate a booking, please provide the following **4 required pieces of information**:
1. **First Name and Surname** (e.g., `Merve Okcu`).
2. **Car Number / License Plate** (e.g., `34ABC123` — alphanumeric uppercase format without spaces).
3. **Reservation Period** (start date/time and end date/time, e.g., `2026-10-20 09:00 to 2026-10-20 18:00`).
4. **Preferred Zone** (optional — defaults to `Zone A Standard` if not specified; `Zone B EV` and `Zone C VIP` are also available).

### Step 2: Human Administrator Confirmation (Human-in-the-Loop)
For security and capacity verification, every reservation request collected by the chatbot is automatically escalated to an **AeroPark Human Administrator**.
- Your reservation status begins as **`PENDING_ADMIN_APPROVAL`**.
- The administrator reviews real-time capacity and vehicle details and either **Approves (`CONFIRMED`)** or **Refuses (`REJECTED`)** the request.

### Step 3: Official Recording & Gate Access
Once the human administrator approves your request:
- Our automated **MCP Reservation Server** permanently records your booking in the official confirmation ledger (`Name | Car Number | Reservation Period | Approval Time`).
- The entry gate cameras will automatically recognize your car number (`Car Number`) 15 minutes before your reservation period begins.

## 2. Cancellation, Grace Period & Refund Policy
- **Grace Period:** You have a **15-minute grace period** before and after your reserved window at no extra charge.
- **Free Cancellation:** Reservations may be cancelled free of charge up to **2 hours before** the scheduled start time.
- **Late Stays:** Vehicles remaining in the garage beyond their reserved period without an extension are billed at the standard hourly rate plus a $10.00 administrative fee.

## 3. Frequently Asked Questions (FAQ)
- **Q: Do I need a paper ticket at the gate?**
  - **A:** No! Our License Plate Recognition (LPR) cameras read your `Car Number` automatically once your reservation is approved by the administrator.
- **Q: Can I check live available spaces and current prices before booking?**
  - **A:** Yes! Ask the chatbot *"How many spaces are available right now?"* or *"What are the current parking prices?"* and it will query our live SQL database for real-time numbers.

