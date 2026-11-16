// Privacy & Consent Information Page
export default function PrivacyPage() {
  return (
    <div style={{ maxWidth: 800, margin: '2rem auto', padding: '0 1.5rem', lineHeight: 1.7 }}>
      <h1 style={{ color: '#a78bfa', marginBottom: 8 }}>⚛ Q-CARE — Privacy & Consent Policy</h1>
      <p style={{ color: '#ff6b6b', background: '#1a0000', padding: '10px 14px', borderRadius: 6, fontSize: '0.9rem' }}>
        ⚠ <strong>Research Prototype Disclaimer:</strong> Q-CARE is an experimental research system.
        It is NOT a medical device, clinical tool, or validated diagnostic system.
        Results must not be used for medical or developmental diagnoses.
      </p>

      {[
        {
          title: '1. What We Collect',
          content: `Q-CARE collects the following information locally on your device:
• Guardian account information (name, email, hashed password — password is never stored in plaintext)
• Child profile information (display name, age category, optional date of birth)
• Video recordings uploaded for assessment
• Extracted video frames and computed feature data
• Assessment session records and results
• Audit logs of authentication events

We do not collect: legal names of children, government IDs, biometric identity databases, or any data transmitted to external servers.`,
        },
        {
          title: '2. Why Video Processing Is Performed',
          content: `Video recordings are processed to extract behavioural features related to screen-time-associated risk indicators. Processing involves:
• Frame extraction and participant tracking (using the guardian-selected bounding box)
• CNN-based visual feature extraction
• LSTM temporal modelling
• QAOA (Quantum Approximate Optimisation Algorithm) feature selection
• VQC (Variational Quantum Circuit) classification
• Age-appropriate risk category output

The system does not perform facial recognition or build a biometric identity database.`,
        },
        {
          title: '3. What Is Stored Locally',
          content: `All data is stored locally on the host device in a SQLite database and local file system. No data is sent to external cloud services.

Stored data includes:
• Guardian account records
• Child profile records
• Assessment session metadata
• Model configurations and training results
• Audit logs

Raw video files are deleted after processing by default (unless retain_video consent is given).`,
        },
        {
          title: '4. How Long Data Is Retained',
          content: `Data is retained until the guardian explicitly deletes it. You may:
• Deactivate a child profile (hides it but preserves records)
• Permanently delete a child profile and all associated records
• Delete individual assessment records
• Request deletion of all guardian account data

There is no automatic data expiry. The guardian controls all retention.`,
        },
        {
          title: '5. Use of Data for Research Training',
          content: `De-identified data may optionally be used for model training and research improvement IF the guardian explicitly consents (separate consent checkbox). This consent:
• Is optional and not required to use Q-CARE
• Can be withdrawn at any time
• Does not apply to raw video files or identifiable information
• Only applies to de-identified feature vectors and labels`,
        },
        {
          title: '6. Consent Choices',
          content: `Q-CARE uses separate consent choices for each purpose:

1. Video assessment — required to use Q-CARE at all
2. Retention of raw video files — optional (default: delete after processing)
3. Use of de-identified data for research training — optional
4. Optional biometric research features — optional (disabled by default; requires separate configuration)

Consent for one purpose does not automatically authorize another purpose.`,
        },
        {
          title: '7. How to Withdraw Consent or Delete Data',
          content: `To withdraw consent or delete data:
• Log in to your guardian account
• Navigate to the child profile
• Use "Deactivate Profile" to hide the profile
• Use "Permanently Delete All Records" to remove all data for a child
• To delete your guardian account, contact the system administrator

After permanent deletion, data cannot be recovered.`,
        },
        {
          title: '8. Child Safety Commitments',
          content: `Q-CARE is designed with the following child safety commitments:
• The age category is ALWAYS selected explicitly by the guardian — it is never inferred from facial appearance or video analysis
• The selected participant is tracked throughout the video — the system will never silently switch to a different person
• No child may independently create a profile or initiate an assessment
• No child profile data is accessible by other guardian accounts
• No biometric identity database is created for children
• A visible recording indicator is shown during all video capture
• Guardians can stop recording immediately at any time`,
        },
        {
          title: '9. Applicable Privacy Requirements',
          content: `Before deploying Q-CARE in any context involving real children, the operator must review and comply with applicable privacy and child-data protection requirements in their jurisdiction, including but not limited to:
• COPPA (Children's Online Privacy Protection Act, USA)
• GDPR and applicable national implementations (EU/EEA)
• DPDPA (Digital Personal Data Protection Act, India)
• Local institutional ethics approval for research involving minors

Q-CARE in its current form is a local research prototype. The operators are responsible for ensuring compliance before any deployment involving real participants.`,
        },
      ].map(({ title, content }) => (
        <div key={title} style={{ background: '#1e1e2e', borderRadius: 10, padding: '1.25rem 1.5rem', marginBottom: 16 }}>
          <h3 style={{ color: '#c4b5fd', marginTop: 0, marginBottom: 10, fontSize: '1rem' }}>{title}</h3>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', color: '#a0a0c0', fontSize: '0.88rem', fontFamily: 'inherit', lineHeight: 1.6 }}>
            {content}
          </pre>
        </div>
      ))}

      <div style={{ background: '#1a1a0e', borderRadius: 8, padding: '12px 16px', borderLeft: '3px solid #d97706', marginBottom: 32 }}>
        <p style={{ margin: 0, color: '#fbbf24', fontSize: '0.85rem' }}>
          Version 1.0 — Q-CARE Research Prototype · All data stored locally · Not a clinical tool
        </p>
      </div>
    </div>
  );
}
