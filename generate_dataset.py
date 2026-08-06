import pandas as pd
import random

# Core high-quality spam examples
spam_samples = [
    "URGENT! You have won a $1,000 Walmart Gift Card. Claim your reward immediately at http://bit.ly/claim-reward-now",
    "Congratulations! You've been selected to receive a free $500 Amazon Voucher. Call 0800 123 4567 to claim.",
    "WINNER!! As a valued customer, you have won a guaranteed £1000 cash or a 4* holiday in Spain! Call 09061701461 now.",
    "IMPORTANT NOTICE: Your Bank Account has been compromised. Verify your details now at https://secure-bank-login-verify.com to avoid account suspension.",
    "Double your Bitcoins in 24 hours! Guaranteed return of 200%. Send your BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa now!",
    "Get cheap prescription drugs online! Viagra, Cialis, Xanax without doctor prescription. Fast worldwide delivery http://online-rx-deals.com",
    "FINAL WARNING: Your IRS tax payment is overdue. Arrest warrant issued. Call emergency helpline immediately at 1-800-555-0199.",
    "Hot single women in your neighborhood want to meet you tonight! No credit card required. Click here to view profiles: http://meet-singles-now.com",
    "You have (1) pending package delivery from FedEx. Action required! Confirm your address and pay $1.99 fee at http://fedex-package-tracking.info",
    "Earn $5,000 per week working from home! No experience required. Simple copy paste job. Apply now at http://easy-home-cash-jobs.com",
    "Exclusive Limited Time Offer! Get 80% off luxury Swiss watches, Rolex, Omega. Free shipping worldwide! Order now http://replica-luxury-watches.com",
    "ALERT: Suspicious login attempt detected on your PayPal account from Russia. If this was not you, secure your account at http://paypal-security-alert-check.com",
    "Claim your free iPhone 15 Pro Max today! You have been chosen in our annual tech giveaway. Click link to claim now!",
    "Low rate pre-approved personal loan up to $50,000! Zero processing fee. Apply in 2 minutes: http://quick-cash-loans-online.com",
    "Urgent response requested: I am Barrister Eric Mensah representing late client with $10.5M deposit. Reply with full details for transfer share.",
    "Free Casino Chips worth $250! Play online poker, slots and blackjack. Register now and win big jackpot!",
    "Your Apple ID has been locked for security reasons. Verify your identity within 24 hours to prevent permanent termination.",
    "Lose 20 lbs in 14 days with revolutionary miracle keto weight loss pill! Order your free trial bottle today!",
    "Get unlimited access to premium adult movies and private live cams for FREE! Click here to activate your trial membership.",
    "Security Alert: Someone accessed your Netflix account from unknown device. Update password now: http://netflix-account-update-sec.com",
    "You have won a free pass to the International Crypto Summit! Register now to receive 500 free tokens.",
    "Make $500 an hour trading forex and options with automatic AI trading bot. 100% win rate guaranteed!",
    "URGENT: Your mobile line will be disconnected in 2 hours. Top up instantly or call customer support at 09058094567",
    "Selected for $2,000 cash grant from Government relief fund. No repayment needed. Click link to enter details.",
    "Huge discount on designer bags, Gucci, Prada, Louis Vuitton! Up to 90% off clearance sale ending today!",
    "Your credit score dropped by 45 points! Check your free credit score report and fix negative items now.",
    "Congratulations winner! Reply YES to claim your free $100 Target shopping card.",
    "Unclaimed funds notice: State treasury holding $4,500 in your name. Verify identity to claim your check.",
    "WARNING: Virus detected on your Computer/Mobile! System compromised. Download antivirus repair tool immediately.",
    "Cheap flight tickets and hotel deals! Book now and get 70% discount on summer vacation packages.",
]

# Core high-quality ham (legitimate) examples
ham_samples = [
    "Hi Alex, please find attached the quarterly financial report for Q3. Let me know if you have any questions before the meeting.",
    "Hey! Are we still meeting for lunch today at 12:30 PM at the Italian place downtown?",
    "Dear Team, Please remember to submit your weekly timesheets by 5 PM today. Thank you for your cooperation.",
    "Thanks for sending the presentation slides. I reviewed them and they look great. Just left a couple of minor comments.",
    "Hi mom, I just landed safely at the airport. Catching a taxi to the hotel now. Will call you once I check in.",
    "Reminder: Your doctor appointment with Dr. Smith is scheduled for tomorrow at 10:00 AM. Reply C to confirm.",
    "Can you please send me the source code repository link for the new frontend dashboard feature?",
    "Good morning, here is the agenda for our project sync call at 2 PM today: 1. Sprint status 2. API integration 3. QA updates.",
    "Hey Dave, happy birthday! Hope you have a wonderful day celebrating with family and friends.",
    "Your order #94821 has been shipped via UPS. Expected delivery date is Thursday, Aug 8th. Track package here.",
    "Hi Sarah, could you please review the attached contract draft and let me know if any clauses need revision?",
    "The meeting has been rescheduled to Friday 11:00 AM due to room availability. Updated calendar invite attached.",
    "Hi everyone, the office will remain closed on Monday for the national holiday. Have a great long weekend!",
    "Thank you for interviewing with our team yesterday. We were impressed with your experience and will follow up by Friday.",
    "Hey, do you happen to have the notes from yesterday's database design workshop?",
    "Hi John, I updated the pull request based on your feedback. Please take another look when you get a chance.",
    "Your monthly electricity bill for July ($84.20) is ready to view online. Auto-pay will process on Aug 15.",
    "Please find attached the receipt for your recent purchase at Best Buy. Thank you for shopping with us.",
    "Hey, what time are we heading to the gym tonight? Let me know if 6 PM works for you.",
    "Dear Student, Your registration for Fall 2026 courses has been confirmed. View your schedule in the student portal.",
    "Hi Mark, I have uploaded the project assets to Google Drive. Let me know if you need any additional formats.",
    "The server maintenance has been completed successfully. All services are up and running smoothly.",
    "Hi, I won't be able to attend today's standup as I'm taking a personal day off. I'll post my updates in Slack.",
    "Thanks for bringing pizza to the team gathering yesterday! It was really delicious.",
    "Hi team, code freeze for release v2.4 starts tonight at midnight. Please merge all approved PRs before then.",
    "Hey, do you want to grab coffee before the client presentation starts?",
    "Dear Customer, Your monthly bank statement for account ending in 4821 is now available in online banking.",
    "Hi Jennifer, can you send over the final invoice for the design work completed last month?",
    "Hey, I left my laptop charger in conference room B. Could you check if it's still on the table?",
    "Hi Team, great job on reaching our Q2 milestone early! Thank you all for your hard work and dedication."
]

# Expand dataset with variations to form a robust 600+ sample dataset for ML training
random.seed(42)

dataset = []

# Generate variations for spam
spam_templates = [
    "URGENT: {action} to win {prize}! Click {link} or call {phone}",
    "Congratulations! You won {prize}. Redeem at {link} now before offer expires.",
    "ALERT: Your {account} account is blocked. Verify at {link} immediately.",
    "Get cheap {product} with free shipping worldwide! Order at {link}",
    "Earn ${amount} daily working from home. No experience needed. Details: {link}",
    "Claim your free ${amount} gift card today! Click {link} to activate.",
    "FINAL NOTICE: Overdue payment for {account}. Call {phone} to prevent legal action.",
    "Exclusive deal! {discount}% off on luxury {product}. Shop now at {link}"
]

prizes = ["$1000 Cash", "iPhone 15 Pro", "$500 Amazon Gift Card", "Free Rolex Watch", "Tesla Model 3", "200 Free Bitcoin"]
links = ["http://claim-now-win.com", "http://secure-verify-account.net", "http://cheap-deals-online.org", "http://bit.ly/exclusive-bonus"]
phones = ["0800-123-9988", "1-800-555-0199", "0906-778-900", "0871-240-0010"]
accounts = ["PayPal", "Bank of America", "Netflix", "Amazon", "Apple ID", "Wells Fargo"]
products = ["Viagra & Cialis", "Designer Watches", "Keto Weight Loss Pills", "Crypto Trading Bot", "Ray-Ban Sunglasses"]

for s in spam_samples:
    dataset.append({"Category": "spam", "Message": s})

for i in range(250):
    tmpl = random.choice(spam_templates)
    msg = tmpl.format(
        action=random.choice(["Act now", "Claim immediately", "Verify your details", "Reply YES"]),
        prize=random.choice(prizes),
        link=random.choice(links),
        phone=random.choice(phones),
        account=random.choice(accounts),
        product=random.choice(products),
        amount=random.choice(["500", "1,000", "2,500", "5,000"]),
        discount=random.choice(["70", "80", "90", "95"])
    )
    dataset.append({"Category": "spam", "Message": msg})

# Generate variations for ham
ham_templates = [
    "Hi {name}, can you review the {doc} by {time}? Let me know your thoughts.",
    "Hey {name}, are we still meeting at {time} for {event}?",
    "Reminder: {event} is scheduled for {day} at {time}. See you there!",
    "Hi team, please find attached the {doc} for our project sync.",
    "Thanks {name} for sending over the {doc}. It looks great!",
    "Hi {name}, could you please help me with the {task} when you have a moment?",
    "Your order for {product} has been confirmed. Track your delivery online.",
    "Hi all, please note that the office will be closed on {day}."
]

names = ["Alex", "Sarah", "John", "David", "Emily", "Michael", "Jessica", "Daniel", "Laura", "Kevin"]
docs = ["quarterly report", "project proposal", "design mockups", "sprint plan", "financial summary", "meeting notes"]
times = ["10:00 AM", "2:00 PM", "4:30 PM", "tomorrow morning", "this afternoon"]
events = ["lunch", "team standup", "client demo", "code review session", "coffee break"]
days = ["Monday", "Wednesday", "Friday", "next Tuesday"]
tasks = ["API bug fix", "database migration", "UI alignment", "unit test coverage", "documentation update"]

for h in ham_samples:
    dataset.append({"Category": "ham", "Message": h})

for i in range(300):
    tmpl = random.choice(ham_templates)
    msg = tmpl.format(
        name=random.choice(names),
        doc=random.choice(docs),
        time=random.choice(times),
        event=random.choice(events),
        day=random.choice(days),
        task=random.choice(tasks),
        product=random.choice(["book", "laptop stand", "monitor", "coffee mug", "keyboard"])
    )
    dataset.append({"Category": "ham", "Message": msg})

df = pd.DataFrame(dataset)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("C:/Users/Jyoshna/.gemini/antigravity/scratch/spam_email_detector/spam.csv", index=False)
print(f"Generated dataset with {len(df)} samples ({sum(df['Category']=='spam')} spam, {sum(df['Category']=='ham')} ham).")
